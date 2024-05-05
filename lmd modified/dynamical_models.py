import numpy as np
import pysindy as ps
import itertools
import torch
from sindy_utils import library_size

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

    
####################################################################################

def get_dynamical_system(name, args=None, normalization=None, use_sine=False):
    systems = {
        'rossler': RosslerSystem,
        'predator_prey': PredatorPrey,
    }

    if name in systems:
        system = systems[name](args, normalization)
        system.get_system()
        return system.f, system.Xi, system.dim, system.z0_mean_sug, system.z0_std_sug
    else:
        raise ValueError(f"Unknown dynamical system: {name}")

####################################################################################

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

####################################################################################

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

####################################################################################


class RosslerSystem(DynamicalSystem):
    def __init__(self, args=None, normalization=None, poly_order=2):
        args = [0.15, 0.2, 10] if args is None else np.array(args)
        self.poly_order = poly_order
        super().__init__(args, normalization)
        self.f = lambda z, t, args = [0]: self.equation(z, t, params=self.args)
        
    def equation(self, Z, t, params=[0.15, 0.2, 10]):
        x, y, z = Z
        a, b, c = params
        return [-y-z, x+a*y, b+z*(x-c)]
    

    def get_system(self):


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

####################################################################################

class PredatorPrey(DynamicalSystem):
    def __init__(self, args=None, normalization=None, poly_order=1):
        args = [1.0, 0.1, 1.5, 0.75] if args is None else np.array(args)
        self.poly_order = poly_order
        super().__init__(args, normalization)
        self.f = lambda z, t, args = [0]: self.equation(z, t, params=self.args)

    def equation(self, Z, t, params=[1, 0.1, 1.5, 0.75]):
        x, y = Z
        a, b, c, d = params
        return [a*x - b*x*y, -c*y + b*d*x*y] 

    def get_system(self):
        self.dim = 2
        n = self.normalization if self.normalization is not None else np.ones((self.dim,))
        self.z0_mean_sug = [10, 5]
        self.z0_std_sug = [8, 8]

        self.Xi = np.zeros((library_size(self.dim, self.poly_order), self.dim))
        exponents = get_poly_exponents(self.poly_order, self.dim)

        for index, exponent in enumerate(exponents):
            if exponent == (1, 0):
                self.Xi[index, 0] = self.args[0]
            elif exponent == (1, 1):
                self.Xi[index, 0] = self.args[1] * n[0]/n[1]
            elif exponent == (0, 1):
                self.Xi[index, 1] = -self.args[2]
            elif exponent == (1, 1):
                self.Xi[index, 1] = -self.args[1] * self.args[3] * n[1]/n[0]






