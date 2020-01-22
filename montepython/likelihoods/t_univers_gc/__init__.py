import os
from montepython.likelihood_class import Likelihood_prior


class t_univers_gc(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl(self, cosmo, data):

        t_u = cosmo.age()
        #print t_u, ' Gyrs'
	loglkl = -0.5 * (t_u - self.t_u) ** 2 / (self.sigma ** 2)
        return loglkl
