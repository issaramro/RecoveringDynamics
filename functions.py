import torch
import torch.nn as nn
from torchdiffeq import odeint as odeint_torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint



def Rossler(X, t, P):
    a, b, c = P
    x, y, z = X
    return - y - z, x + a*y, b - c*z + x*z

def LV(X, t,P):
    x, y = X
    b1, b2, a12, a21 = P
    return b1*x - a12*x*y, a21*x*y - b2*y

def Solution(func, x0, t, P):
    return odeint(func, x0, t, args = (P,)).T


# to be used in MI
def entropy(p):
    p_norm = p / np.sum(p)
    p_norm = p_norm[np.nonzero(p_norm)]
    S = -np.sum(p_norm* np.log(p_norm))  
    return S


# calculates mutual information between X and Y
def MI(X,Y,bins):
    p_XY = np.histogram2d(X,Y,bins)[0]
    p_X = np.histogram(X,bins)[0]
    p_Y = np.histogram(Y,bins)[0]
 
    S_X = entropy(p_X)
    S_Y = entropy(p_Y)
    S_XY = entropy(p_XY)
 
    MI = S_X + S_Y - S_XY
    return MI


# calculates the percentages of the the False Nearest neighbors for a given range of k values
def FNN(xs, k_vals, eps, tau):
    per = []
    n = len(xs)
    for k in k_vals:
        FNN = 0
        NN = 0
        for i in range(n - (k * tau) - 1):
            for j in range(n- (k * tau) - 1):
                if j != i:
                    if (eps - np.linalg.norm(xs[i] - xs[j])) > 0:
                        Rd = 0
                        for o in range(k):
                            Rd = Rd + (xs[i + (o - 1) * tau] - xs[j + (o - 1) * tau])**2
                        D = np.abs(xs[i + k * tau] - xs[j + k * tau])
                        if Rd != 0 and D/(np.sqrt(Rd)) >= 10:
                            FNN += 1
                        else:
                            NN += 1
        per.append((FNN/NN)*100)
    return per



def jacobian(y, x, create_graph=False, retain_graph=True, allow_unused=False):
    jac = []
    for i in range(y.shape[-1]):
        jac_i = torch.autograd.grad(
            y[..., i], x, torch.ones_like(y[..., i]), allow_unused=allow_unused, retain_graph=retain_graph, create_graph=create_graph)[0]
        jac.append(jac_i)
    return torch.stack(jac, dim=-1)


def plot(xs, ys, zs, v1, v2, v3, x_ae, y_ae, z_ae):

    fig, ax = plt.subplots(3, 1, figsize=(8, 9))
    fig_ = plt.figure(figsize=(13, 13))
    ax3d = fig_.add_subplot(1, 3, 1, projection='3d')

    ax[0].plot(xs[:len(x_ae)], label='True', color = "k")
    ax[0].plot(x_ae, label='Transformed',linestyle='dashed', color = "maroon")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("x")
    ax[0].legend()
    ax[1].plot(ys[:len(y_ae)], label='True', color = "k")
    ax[1].plot(y_ae, label='Transformed',linestyle='dashed', color = "maroon")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("y")
    ax[1].legend()
    ax[2].plot(zs[:len(z_ae)], label='True', color = 'k')
    ax[2].plot(z_ae, label='Transformed', linestyle='dashed', color = "maroon")
    ax[2].set_xlabel("t")
    ax[2].set_ylabel("z")
    ax[2].legend()
    ax3d.plot(x_ae, y_ae, z_ae, lw=.3,label='Transformed', color ='maroon')
    ax3d.plot(xs, ys, zs, lw=.3,label='True', color= 'k')
    ax3d.plot(v1, v2, v3, lw=.3, color = 'r', label ="Embedded")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.legend()
    plt.title("The reconstructed Rossler attractor")
    plt.tight_layout()
    plt.show()
    fig3, ax3 = plt.subplots(1, 3, figsize=(10, 9), subplot_kw={'projection': '3d'})
    ax3[0].plot(xs, ys, zs, label="True", color = "k")
    ax3[1].plot(v1, v2, v3, label="Embedded", color = "k")
    ax3[2].plot(x_ae, y_ae, z_ae, label="Transformed",color = "k")
    ax3[0].legend()
    ax3[1].legend()
    ax3[2].legend()
    plt.show()