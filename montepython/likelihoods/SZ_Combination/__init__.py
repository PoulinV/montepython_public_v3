# This is a prior on the linear combination of tSZ and kSZ contributions to the 
# observed Cls, following https://arxiv.org/pdf/1907.12875 (see Eq. 23)
# Also: https://arxiv.org/pdf/1507.02704.pdf
# https://arxiv.org/pdf/1502.01589.pdf
import os
from montepython.likelihood_class import Likelihood_prior

class SZ_Combination(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl(self, cosmo, data):
        A_tsz = 0.
	if 'A_sz' in self.use_nuisance:
	    A_tsz = data.mcmc_parameters['A_sz']['current']*data.mcmc_parameters['A_sz']['scale']
        A_ksz = 0.
	if 'ksz_norm' in self.use_nuisance:
	    A_ksz = data.mcmc_parameters['ksz_norm']['current']*data.mcmc_parameters['ksz_norm']['scale']

        # The linear combination that is constrained
        A_SZ = A_ksz + 1.6*A_tsz
        loglkl = -0.5 * (A_SZ - self.A_SZ_mean) ** 2 / (self.A_SZ_sigma ** 2)
        return loglkl
