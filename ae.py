import torch
import torch.nn as nn
from torchdiffeq import odeint as odeint_torch
import numpy as np
from functions import jacobian

### Autoencoder to be used for these cases: ###
# Supervised case: Supervised = True, x_true is not None, and all ls = 0
# Known system (Rossler case): RosslerSystem = True, l1 = 1e-1, l2 = 1e-1, l_rossler = 1e-2, l4 = 1e-2, (a,b,c) get updated
# Unknown system - Autoencoder then SINDy: l1 = 1, l2 = 1
# Unknown system - SINDy Autoencoder: l1 = 1, l2 = 5*1e-3, l3 = 1e-6, l4 = 1e-6, l5 = 1, l6 = ?, coefficents and theta get updated
class Autoencoder(nn.Module):
    def __init__(self, tau, x_true = None, RosslerSystem = False, Supervised = False, input_dim = 3, 
                 latent_dim = 3, library_dim = 10, state_dim = 3, poly_order = 2,
                 l={'l1': 1, 'l2': 5*1e-3, 'l3': 1e-6, 'l4': 1e-6, 'l5': 1, 'l6': 0, 'l1_rossler': 0, 'l2_rossler':0}):
        super(Autoencoder, self).__init__()

        self.tau = tau
        self.x_true = x_true
        self.RosslerSystem = RosslerSystem
        self.Supervised = Supervised
        self.library_dim = library_dim
        self.state_dim = state_dim
        self.poly_order = poly_order
        self.n_dimensions = state_dim
        self.degree = poly_order
        self.l = l
        if self.RosslerSystem:
            self.a = nn.Parameter(torch.tensor(1.1,requires_grad=True, dtype = torch.float32))
            self.b = nn.Parameter(torch.tensor(0.2,requires_grad=True, dtype = torch.float32))
            self.c = nn.Parameter(torch.tensor(10,requires_grad=True, dtype = torch.float32))
        else:
            self.coefficients = nn.Parameter(torch.randn(library_dim, state_dim))

        n = 100
        self.encoder = nn.Sequential(
        nn.Linear(input_dim, n),
        nn.ReLU(),
        nn.Linear(n, 40),
        nn.ReLU(),
        nn.Linear(40, 10),
        nn.ReLU(),
        nn.Linear(10, latent_dim) 
        )

        self.decoder = nn.Sequential(
        nn.Linear(latent_dim, 10),
        nn.ReLU(),
        nn.Linear(10, 40),
        nn.ReLU(),
        nn.Linear(40, n),
        nn.ReLU(),
        nn.Linear(n, input_dim) 
        )

    def decode(self, z):
        decoded = self.decoder(z)
        return decoded
    
    def encode(self, z):
        encoded = self.encoder(z)
        return encoded
    
    def rossler(self, X):
        x, y, z = X[:,0], X[:,1], X[:,2]
        dx = - y - z
        dy = x + self.a * y
        dz = self.b - self.c * z + x * z
        return torch.stack([dx, dy, dz], dim=1)
    
    def phi(self, x):
        library = [torch.ones(x.size(0), 1)]
        for i in range(self.state_dim):
            library.append(x[:, i:i+1])  

        if self.poly_order == 2:
            for i in range(self.state_dim):
                for j in range(i, self.state_dim):
                    library.append(x[:, i:i+1] * x[:, j:j+1])

        return torch.cat(library, dim=1)
    
    
    def SINDy_num(self, t, x):
        dxdt = torch.matmul(self.phi(x), self.coefficients)
        return dxdt 
    
    def integrate(self, x0, t):
        try:
            x_p = odeint_torch(self.SINDy_num, x0, t) 
        except AssertionError as error:
            print(error)
            return None 
        return x_p
    
    
    def loss(self, v, dvdt, criterion):
        with torch.autograd.enable_grad():
            x = self.encoder(v)
            v_bar = self.decoder(x)

        time = torch.tensor(np.linspace(0, (np.pi/100)*len(x), len(x), endpoint=False),requires_grad=False, dtype=torch.float32)
        loss1 = criterion(v_bar, v)*self.l['l1']
        loss2 = criterion(x[:, 0], v[:, 0])*self.l['l2']
        loss = loss1 + loss2

        if self.l['l3'] > 0 or self.l['l4'] > 0:

            dxdt_SINDy = self.SINDy_num(time, x)

            with torch.autograd.enable_grad():
                dxdv = jacobian(x, v)
            dxdt = torch.einsum('ijk,ij->ik', dxdv, dvdt)

            with torch.autograd.enable_grad():
                dvdx = jacobian(v_bar, x)
            dvdt_dec = torch.einsum('ijk,ij->ik', dvdx, dxdt_SINDy)

            loss3 = criterion(dxdt, dxdt_SINDy)*self.l['l3']
            loss4 = criterion(dvdt_dec, dvdt)*self.l['l4']

            loss += (loss3 + loss4)

        if self.l['l5'] > 0:
            loss5 = torch.norm(self.coefficients, 1)*self.l['l5']
            loss += loss5

        if self.l['l6'] > 0:
            x_pred = self.integrate(x[0].view(-1,3), time)
            if x_pred is not None:  
                x_first = x_pred[:,0]
                n = len(x_first)
                x1 = x_first[0: n - 2*self.tau]
                x2 = x_first[self.tau: n - self.tau]
                x3 = x_first[2*self.tau: n]
                x_delayed = [x1, x2, x3]
                v_delayed = [v[:,0][:n], v[:,1][:n], v[:,2][:n]]
                loss6 = criterion(x_delayed, v_delayed)*self.l['l6']
                loss += loss6
            else:
                loss += 0

        if self.RosslerSystem:
            rossler_vals = self.rossler(x)
            with torch.autograd.enable_grad():
                dxdv = jacobian(x, v)
            dxdt = torch.einsum('ijk,ij->ik', dxdv, dvdt)
            
            with torch.autograd.enable_grad():
                dvdx = jacobian(v_bar, x)
            dvdt_dec = torch.einsum('ijk,ij->ik', dvdx, rossler_vals)

            loss_rossler1 = criterion(rossler_vals, dxdt)*self.l['l1_rossler']
            loss_rossler2 = criterion(dvdt_dec, dvdt)*self.l['l2_rossler']
            loss += (loss_rossler1 + loss_rossler2)

        if self.Supervised:
            loss_supervised = criterion(self.x_true, x)
            loss += loss_supervised

        return loss
    
