This repository contains the likelihood module for the KiDS+VIKING-450 (in short: KV450) correlation function measurements from [Hildebrandt et al. 2018 (arXiv:1812.06076)](http://adsabs.harvard.edu/abs/2018arXiv181206076H).
The module will be working 'out-of-the-box' within a [MontePython](https://github.com/brinckmann/montepython_public) and [CLASS](https://github.com/lesgourg/class_public) (version >= 2.6 and including the HMcode module) setup. The required KiDS+VIKING-450 data files can be downloaded from the [KiDS science data webpage](http://kids.strw.leidenuniv.nl/sciencedata.php) and the parameter file for reproducing the fiducial run of [Hildebrandt et al. 2018 (arXiv:1812.06076)](http://adsabs.harvard.edu/abs/2018arXiv181206076H) is supplied in the subfolder `INPUT` within this repository.

TB comment: as of MontePython v3.2 release date HMcode is still not publicly available in CLASS, write to e.g. sbrieden AT icc.ub.edu and thejs.brinckmann AT stonybrook.edu for access.

Assuming that MontePython (with CLASS version >= 2.6 and including the HMcode module) is set up (we recommend to use the MultiNest sampler!), please proceed as follows:

1) Clone this repository

`git clone https://github.com/fkoehlin/kv450_cf_likelihood_public.git`

2) Copy `__init__.py` and `kv450_cf_likelihood_public.data` from this repository into a folder named `kv450_cf_likelihood_public` within `/your/path/to/montepython_public/montepython/likelihoods/`.

(you can rename the folder to whatever you like, but you must use this name then consistently for the whole likelihood which implies to rename the `*.data`-file, including the prefixes of the parameters defined in there, the name of the likelihood in the `__init__.py`-file and also in the `*.param`-file.)

3) Set the path to the data folder (i.e. `KV450_COSMIC_SHEAR_DATA_RELEASE` from the tarball available from the [KiDS science data webpage](http://kids.strw.leidenuniv.nl/sciencedata.php') in `kv450_cf_likelihood_public.data` and modify parameters as you please (note that everything is set up to reproduce the fiducial run with `kv450_cf.param`).

4) Start your runs using e.g. the `kv450_cf.param` supplied in the subfolder `INPUT` within this repository.

5) Contribute your developments/bugfixes to this likelihood (please use a dedicated branch per fix/feature).

6) If you publish your results based on using this likelihood, please cite [Hildebrandt et al. 2018 (arXiv:1812.06076)](http://adsabs.harvard.edu/abs/2018arXiv181206076H) and all further references for the KiDS+VIKING-450 data release (as listed on the [KiDS science data webpage](http://kids.strw.leidenuniv.nl/sciencedata.php)) and also all relevant references for Monte Python and CLASS.

Refer to `run_with_multinest.sh` within the subfolder `INPUT` for all MultiNest-related settings that were used for the fiducial runs.

Note when you run the likelihood for the very first time, the covariance matrix from the data release (given in list format) needs to be converted into an actual NxN matrix format. This will take several minutes, but only once. The reformatted matrix will be saved to and loaded for all subsequent runs of the likelihood from the folder `FOR_MONTE_PYTHON` within the main folder `KV450_COSMIC_SHEAR_DATA_RELEASE` of the data release.

WARNING: This likelihood only produces valid results for `\Omega_k = 0`, i.e. flat cosmologies!

For questions/comments please use the issue-tracking system!
