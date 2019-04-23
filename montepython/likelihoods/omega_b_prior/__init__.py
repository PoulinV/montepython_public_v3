import os
from montepython.likelihood_class import Likelihood_prior

class omega_b_prior(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl(self, cosmo, data):

        omega_b_h2 = cosmo.omega_b()

        loglkl = -0.5 * (omega_b_h2 - self.omega_b_h2) ** 2 / (self.sigma ** 2)

        return loglkl
