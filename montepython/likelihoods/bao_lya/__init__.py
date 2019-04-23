import os
import numpy as np
from montepython.likelihood_class import Likelihood
import scipy.constants as const

class bao_lya(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # define array for values of z and data points
        self.corr_types = []
        self.z = np.array([], 'float64')
        self.types = []

        scan_locations = {}
        scan_locations['cf'] = self.data_directory + '/' + self.cf_scan
        scan_locations['xcf'] = self.data_directory + '/' + self.xcf_scan

        # read redshifts and data points
        for line in open(os.path.join(
                self.data_directory, self.file), 'r'):
            if (line.strip().find('#') == -1) and (len(line.strip())>0):
                self.corr_types += [line.split()[0]]
                self.z = np.append(self.z, float(line.split()[1]))
                self.types += [set([int(line.split()[2]),float(line.split()[3])])]

        # number of data points
        self.num_points = np.shape(self.z)[0]

        #Make our interpolators
        self.chi2_interpolators = chi2_interpolators(scan_locations,self.boss_da_over_rd_fid,self.boss_c_over_Hrd_fid)

        # end of initialization

    # compute log likelihood
    def loglkl(self, cosmo, data):

        chi2 = 0.

        # for each point, compute angular distance da, radial distance dr,
        # volume distance dv, sound horizon at baryon drag rs_d,
        # theoretical prediction and chi2 contribution
        # classes: (D_V/rs=3, Dv/Mpc=4, DA/rs=5, c/Hrs=6, rs/D_v=7, D_M/rs=8, H rs/rs_fid=9, D_M rs_fid/rs=10)
        for i in range(self.num_points):

            da = cosmo.angular_distance(self.z[i])
            dr = self.z[i] / cosmo.Hubble(self.z[i])
            H  = cosmo.Hubble(self.z[i]) * const.c / 1000.

            dv = pow(da * da * (1 + self.z[i]) * (1 + self.z[i]) * dr, 1. / 3.)
            dm = da * (1 + self.z[i])

            rd = cosmo.rs_drag() * self.rd_rescale

            if (self.types[i] == set([5,6])):
                da_over_rd = da / rd
                c_over_Hrd = (const.c / 1000.) / (H * rd)
                chi2 += self.chi2_interpolators.get_chi2_distances(da_over_rd,c_over_Hrd,corr_type=self.corr_types[i])
            else:
                raise io_mp.LikelihoodError(
                    "In likelihood %s. " % self.name +
                    "BAO data types %s " % self.types[i] +
                    "in %d-th line not appropriate for BOSS DR11" % i)

        # return ln(L)
        lkl = - 0.5 * chi2

        return lkl

#Class to read alpha_t by alpha_p chi2 scans e.g. from BOSS and interpolate.
from scipy.interpolate import RectBivariateSpline
class chi2_interpolators():
    def __init__(self,scan_locations,DA_over_rd_fid,c_over_Hrd_fid):
        """
        Arguments:
        scan_locations: dictionary of filepaths to the different scans, with
                        keys as scan types.
        DA_over_rd_fid: fiducial value of DA/rd used to calculate alpha_t.
        c_over_Hrd_fid: fiducial value of c/Hrd used to calculate alpha_p.
        """

        #Create a dictionary containing an interpolator for each scan.
        interpolators = {}
        for corr_type in scan_locations:
            scan = np.loadtxt(scan_locations[corr_type])

            at = np.array(sorted(set(scan[:,0])))
            ap = np.array(sorted(set(scan[:,1])))

            N_at = at.shape[0]
            N_ap = ap.shape[0]
            grid = np.zeros((N_at,N_ap))

            for i in range(N_ap):
                #Filter the data to only those corresponding to the ap value.
                indices = (scan[:,1]==ap[i])
                scan_chunk = scan[indices,:]
                #Ensure that they're sorted by at value.
                scan_chunk = scan_chunk[scan_chunk[:,0].argsort()]
                #Add the chi2 column to the grid.
                #Note that the grid is of shape (N_at,N_ap)
                grid[:,i] = scan_chunk[:,2]

            #Make the interpolator (x refers to at, y refers to ap).
            interpolators[corr_type] = RectBivariateSpline(at,ap,grid,kx=1,ky=1)

        #Add the dictionary to the object.
        self.interpolators = interpolators
        self.DA_over_rd_fid = DA_over_rd_fid
        self.c_over_Hrd_fid = c_over_Hrd_fid

        return

    #Function to return the interpolated value of chi2 given distance measures.
    def get_chi2_distances(self,DA_over_rd,c_over_Hrd,corr_type='cf'):
        """
        Arguments:
        DA_over_rd_fid: value of DA/rd to evaluate chi2 for.
        c_over_Hrd_fit: value of c/Hrd to evaluate chi2 for.
        corr_type:      which scan to interpolate.

        Returns:
        chi2:           value of chi2
        """

        #Convert distances to alphas.
        at = DA_over_rd/self.DA_over_rd_fid
        ap = c_over_Hrd/self.c_over_Hrd_fid

        #With the new alphas, get the log likelihood.
        chi2 = self.interpolators[corr_type](at,ap)

        return chi2
