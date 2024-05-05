import torch
import torch.nn as nn
from torchdiffeq import odeint as odeint_torch
import numpy as np
from functions import jacobian
import pysindy as ps
from scipy.special import binom
import itertools
import torch.nn.functional as F

### Autoencoder to be used for these cases: ###
# Supervised case: Supervised = True, x_true is not None, and all ls = 0
# Known system (Rossler case): RosslerSystem = True, l1 = 1e-1, l2 = 1e-1, l_rossler = 1e-2, l4 = 1e-2, (a,b,c) get updated
# Unknown system - Autoencoder then SINDy: l1 = 1, l2 = 1
# Unknown system - SINDy Autoencoder: l1 = 1, l2 = 5*1e-3, l3 = 1e-6, l4 = 1e-6, l5 = 1, l6 = ?, coefficents and theta get updated
class Autoencoder(nn.Module):
    def __init__(self, tau, loss_fn, ode_model, optim_enc, optim_ode, encoder, decoder, x_true = None, true_coef = None, rossler_system = False, lv_system = False, supervised = False,
                 l={'l1': 1, 'l2': 5*1e-3, 'l3': 1e-6, 'l4': 1e-6, 'l5': 1, 'l6': 0, 'l1_system': 0, 'l2_system':0}):
        super(Autoencoder, self).__init__()

        self.tau = tau
        self.x_true = x_true
        self.true_coef = true_coef
        self.rossler_system = rossler_system
        self.lv_system = lv_system
        self.supervised = supervised
        self.ode_model = ode_model
        self.encoder = encoder
        self.decoder = decoder
        self.optim_enc = optim_enc
        self.optim_ode = optim_ode
        self.loss_fn = loss_fn
        self.l = l
    
    def loss(self, v, dvdt):
        with torch.autograd.enable_grad():
            x = self.encoder(v)
            v_bar = self.decoder(x)

        time = torch.tensor(np.linspace(0, (np.pi/100)*len(x), len(x), endpoint=False),requires_grad=False, dtype=torch.float32)
        time_int = torch.tensor(np.linspace(0, (np.pi/100)*50, 50, endpoint=False),requires_grad=False, dtype=torch.float32) # to be used in integration

        loss1 = self.loss_fn(v_bar, v)*self.l['l1']
        loss2 = self.loss_fn(x[:, 0], v[:, 0])*self.l['l2']
        loss = loss1 + loss2

        if self.l['l3'] > 0 or self.l['l4'] > 0:

            dxdt_SINDy = self.ode_model.system.get_dxdt(x)

            with torch.autograd.enable_grad():
                dxdv = jacobian(x, v)
            dxdt = torch.einsum('ijk,ij->ik', dxdv, dvdt)

            with torch.autograd.enable_grad():
                dvdx = jacobian(v_bar, x)
            dvdt_dec = torch.einsum('ijk,ij->ik', dvdx, dxdt_SINDy)

            loss3 = self.loss_fn(dxdt, dxdt_SINDy)*self.l['l3']
            loss4 = self.loss_fn(dvdt_dec, dvdt)*self.l['l4']

            loss += (loss3 + loss4)

        if self.l['l5'] > 0:
            loss5 = torch.norm(self.ode_model.theta, 1)*self.l['l5']
            loss += loss5

        if self.l['l6'] > 0:
            x_pred = self.ode_model(x[0].view(-1,3))
            print(x_pred.shape)
            if x_pred is not None:  
                x_first = x_pred[:, :, 0]
                print(x_first.shape)
                n = len(x_first)
                x1 = x_first[0: n - 2*self.tau]
                x2 = x_first[self.tau: n - self.tau]
                x3 = x_first[2*self.tau: n]
                print(torch.tensor(x1).squeeze().shape, v[:len(x1), 0])
                x_delayed = torch.cat([torch.tensor(x1).squeeze().unsqueeze(1), torch.tensor(x2).squeeze().unsqueeze(1),  torch.tensor(x3).squeeze().unsqueeze(1)], dim=1)
                v_delayed = torch.cat([v[:len(x1), 0].unsqueeze(1), v[:len(x1), 1].unsqueeze(1), v[:len(x1), 2].unsqueeze(1)], dim=1).float()
                print(x_delayed.shape, v_delayed.shape)
                loss6 = self.loss_fn(x_delayed, v_delayed)*self.l['l6']
                loss += loss6
            else:
                loss += 0

        if self.rossler_system:
            rossler_vals = self.rossler(x)
            with torch.autograd.enable_grad():
                dxdv = jacobian(x, v)
            dxdt = torch.einsum('ijk,ij->ik', dxdv, dvdt)
            
            with torch.autograd.enable_grad():
                dvdx = jacobian(v_bar, x)
            dvdt_dec = torch.einsum('ijk,ij->ik', dvdx, rossler_vals)

            loss_rossler1 = self.loss_fn(rossler_vals, dxdt)*self.l['l1_system']
            loss_rossler2 = self.loss_fn(dvdt_dec, dvdt)*self.l['l2_system']
            loss += (loss_rossler1 + loss_rossler2)

        if self.lv_system:
            lv_vals = self.lv(x)
            with torch.autograd.enable_grad():
                dxdv = jacobian(x, v)
            dxdt = torch.einsum('ijk,ij->ik', dxdv, dvdt)
            
            with torch.autograd.enable_grad():
                dvdx = jacobian(v_bar, x)
            dvdt_dec = torch.einsum('ijk,ij->ik', dvdx, lv_vals)

            loss_lv1 = self.loss_fn(lv_vals, dxdt)*self.l['l1_system']
            loss_lv2 = self.loss_fn(dvdt_dec, dvdt)*self.l['l2_system']
            loss += (loss_lv1 + loss_lv2)

        if self.supervised:
            loss_supervised = self.loss_fn(self.x_true, x)
            loss += loss_supervised

        
        return loss
    


class network(nn.Module):
    def __init__(self, layer_dims=[100, 50, 10, 3], activation=F.elu):
        super(network, self).__init__()
        self.act = activation
        self.layer_dims = layer_dims
        self.input_dim = layer_dims[0] 
        self.output_dim = layer_dims[-1] 
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_dims)-1):
            self.layers.append( nn.Linear(self.layer_dims[i], self.layer_dims[i+1]))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        return x

class model_torch(nn.Module):
    def __init__(self, system, params):
        super(model_torch, self).__init__()
        self.params = params
        self.system = system

    def forward(self, t, x):
        dx_dt = self.system(x, t, self.params)
        return torch.stack(dx_dt)

    def get_dxdt(self, x):
        # Assumes torch tensor input
        return self.forward(0, x.T).T

class integrator(nn.Module):
    def __init__(self, n_tsteps, dt, system, init_params=[10, 28, 8/3]):
        super(integrator, self).__init__()
        self.n_tsteps = n_tsteps
        self.init_params = init_params
        self.base_system = system
        self.time = torch.tensor(np.linspace(0, dt*n_tsteps, n_tsteps, endpoint=False), requires_grad=False, dtype=torch.float32)
        self.theta = nn.Parameter(torch.tensor(self.init_params, requires_grad=True, dtype=torch.float32))
        self.system = model_torch(self.base_system, self.theta)
    
    def forward(self, x0):
        try:
            H_pred = odeint_torch(self.system, x0.T, self.time) 
        except AssertionError as error:
            print(f"Caught an error while integrating model: {error}\
                  \n This might mean that the model is numerically unstable.")
            return None 
        if len(x0.shape)==1:
            return H_pred
        else:
            return H_pred.permute(2, 0, 1)
        


class PolySindyModel:
    def __init__(self, n_dimensions, degree=3, mask=None, include_interaction=True, 
                 interaction_only=False, include_bias=True):
        self.n_dimensions = n_dimensions
        self.degree = degree
        self.mask = mask 
        self.include_interaction = include_interaction
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.pslib = ps.PolynomialLibrary(degree=self.degree)
        self.combs = list(self.pslib._combinations(self.n_dimensions, 
                                                                self.degree, 
                                                                self.include_interaction, 
                                                                self.interaction_only, 
                                                                self.include_bias))

    def symb_library(self, Z): 
        Zdot_lib = [] 
        for comb in self.combs:
            term = 1
            for elem in comb:
                term *= Z[elem]
            Zdot_lib.append(term)
        return Zdot_lib

    def num_library_terms(self):
        return len(self.combs)
    
    def symb_library_names(self): 
        varnames = list('xyzwvpqs')
        Zdot_lib = [] 
        for comb in self.combs:
            term = '' 
            for elem in comb:
                term += varnames[elem]
            Zdot_lib.append(term)
        return Zdot_lib

    def fitting_params_shape(self):
        return (self.num_library_terms(), self.n_dimensions)

    def poly_system(self, Z, t, Theta):
        zdot_list = []
        if self.mask is None:
            self.mask = torch.ones(Theta.shape)
        for i in range(self.n_dimensions):
            zidot = sum([term*coef for term, coef, m in zip(self.symb_library(Z), Theta[i, :], self.mask[i, :]) if m!=0])
            zdot_list.append(zidot)
        return zdot_list
    


class DynamicalSystem:
    def __init__(self, args, normalization=None):
        self.args = args
        self.normalization = normalization
        self.f = None
        self.Xi = None
        self.dim = None
        self.z0_mean_sug = None
        self.z0_std_sug = None

    def get_system(self):
        raise NotImplementedError

def get_poly_exponents(poly_order, dimension):
    exponents = []
    for order in range(poly_order + 1):
        current_order_exponents = []
        for exp in itertools.product(*[range(order + 1)] * dimension):
            if sum(exp) == order:
                current_order_exponents.append(exp)
        current_order_exponents.sort(reverse=True)
        exponents.extend(current_order_exponents)
    return exponents


class RosslerSystem(DynamicalSystem):
    def __init__(self, args=None, normalization=None, poly_order=2):
        args = [0.2, 0.2, 5.7] if args is None else np.array(args)
        self.poly_order = poly_order
        super().__init__(args, normalization)

    def equation(self, Z, t, params=(0.2, 0.2, 5.7)):
        x, y, z = Z
        a, b, c = params
        return [-y-z, x+a*y, b+z*(x-c)]

    def get_system(self):
        self.f = lambda z, t: self.equation(z, t, params=self.args)

        self.dim = 3
        n = self.normalization if self.normalization is not None else np.ones((self.dim,))
        self.z0_mean_sug = [0, 1, 0]
        self.z0_std_sug = [2, 2, 2]
        self.Xi = np.zeros((library_size(self.dim, self.poly_order), self.dim))
        exponents = get_poly_exponents(self.poly_order, self.dim)

        for index, exponent in enumerate(exponents):
            if exponent == (0, 0, 0):
                self.Xi[index, 2] = n[2]*self.args[1] 
            elif exponent == (1, 0, 0):
                self.Xi[index, 1] = n[1]/n[0]
            elif exponent == (0, 1, 0):
                self.Xi[index, 0] = -n[0]/n[1] 
                self.Xi[index, 1] = self.args[0]
            elif exponent == (0, 0, 1):
                self.Xi[index, 2] = -self.args[2]
                self.Xi[index, 2] = -n[0]/n[2]  
            elif exponent == (1, 0, 1):
                self.Xi[index, 1] = 1/n[0] 

def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l