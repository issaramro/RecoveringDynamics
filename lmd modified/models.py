from torch import nn
import torch
from torchdiffeq import odeint 
import torch.nn.functional as F
import numpy as np
from functions import jacobian

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
    
    ###########

class model_torch(nn.Module):
    def __init__(self, system, params):
        super(model_torch, self).__init__()
        self.params = params
        self.system = system

    def forward(self, t, x):
        dx_dt = self.system(x, t, self.params)
        return torch.stack(dx_dt)

    def get_dxdt(self, x):
        return self.forward(0, x.T).T


class integrator(nn.Module):
    def __init__(self, n_tsteps, dt, system, init_params=None):
        super(integrator, self).__init__()
        self.n_tsteps = n_tsteps
        self.init_params = init_params
        self.base_system = system
        self.time = torch.tensor(np.linspace(0, dt*n_tsteps, n_tsteps, endpoint=False), requires_grad=False, dtype=torch.float32)
        self.theta = nn.Parameter(torch.tensor(self.init_params, requires_grad=True, dtype=torch.float32))
        self.system = model_torch(self.base_system, self.theta)
    
    def forward(self, x0):
        try:
            x_pred = odeint(self.system, x0.T, self.time) 
        except AssertionError as error:
            print(error)
            return None 
        if len(x0.shape)==1:
            return x_pred
        else:
            return x_pred.permute(2, 0, 1)

    ###########

class Autoencoder:
    def __init__(self, tau,
                 ode_model, 
                 loss_fn, 
                 optim_enc, 
                 optim_ode, 
                 tstep_pred, 
                 encoder=None, 
                 decoder=None,
                 l={'x0':1, 'cons': 1, 'l1': 0, 'recon': 0, 'x_dot': 0, 'v_dot': 0}):

        self.tau = tau
        self.encoder = encoder
        self.decoder = decoder
        self.ode_model = ode_model
        self.loss_fn = loss_fn
        self.tstep_pred = tstep_pred
        self.optim_enc = optim_enc
        self.optim_ode = optim_ode
        self.l = l

    def loss(self, v, dvdt=None):

        with torch.autograd.enable_grad():
            x = self.encoder(v)
            v_bar = self.decoder(x)
        
        loss_x0 = self.loss_fn(x[:, 0], v[:, 0])*self.l['x0']
        loss_recon = self.loss_fn(v_bar, v)*self.l['recon']
        loss = loss_x0 + loss_recon

        if self.l['cons'] > 0:
            if len(x[0]) == 3:
                x_pred = self.ode_model(x[0].view(-1,3))
                if x_pred is not None:
                    x_first = x_pred[:, :, 0]
                    x_first = x_first.squeeze()
                    n = len(x_first)
                    x1 = x_first[0: n - 2*self.tau]
                    x2 = x_first[self.tau: n - self.tau]
                    x3 = x_first[2*self.tau: n]
                    x_delayed = torch.cat([torch.tensor(x1).squeeze().unsqueeze(1), torch.tensor(x2).squeeze().unsqueeze(1), torch.tensor(x3).squeeze().unsqueeze(1)], dim=1)
                    v_delayed = torch.cat([v[:len(x1), 0].unsqueeze(1), v[:len(x1), 1].unsqueeze(1), v[:len(x1), 2].unsqueeze(1)], dim=1).float()

                    loss += self.loss_fn(v_delayed, x_delayed)*self.l['cons']
            elif len(x[0]) == 2:
                x_pred = self.ode_model(x[0].view(-1,2))
                if x_pred is not None:
                    x_first = x_pred[:, :, 0]
                    x_first = x_first.squeeze()
                    n = len(x_first)
                    x1 = x_first[0: n - 2*self.tau]
                    x2 = x_first[self.tau: n - self.tau]
                    x_delayed = torch.cat([torch.tensor(x1).squeeze().unsqueeze(1), torch.tensor(x2).squeeze().unsqueeze(1)], dim=1)
                    v_delayed = torch.cat([v[:len(x1), 0].unsqueeze(1), v[:len(x1), 1].unsqueeze(1)], dim=1).float()

                    loss += self.loss_fn(v_delayed, x_delayed)*self.l['cons']
            else:
                loss += 0

        if self.l['l1'] > 0:
            loss += torch.norm(self.ode_model.theta, 1)*self.l['l1']

        if self.l['x_dot'] > 0:
            dxdt_SINDy = self.ode_model.system.get_dxdt(x)
            with torch.autograd.enable_grad():
                dxdv = jacobian(x, v)
            dxdt = torch.einsum('ijk,ij->ik', dxdv, dvdt)
            
            loss += self.loss_fn(dxdt_SINDy, dxdt)*self.l['x_dot']

        if self.l['v_dot'] > 0:
            with torch.autograd.enable_grad():
                dvdx = jacobian(v_bar, x)
            dvdt_enc = torch.einsum('ijk,ij->ik', dvdx, dxdt_SINDy)
            loss += self.loss_fn(dvdt_enc, dvdt)*self.l['v_dot']

        return loss

   


