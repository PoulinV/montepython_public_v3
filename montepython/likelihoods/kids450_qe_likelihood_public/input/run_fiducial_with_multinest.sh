#!/bin/bash/ 
python /your/path/to/montepython_public/montepython/MontePython.py run \
	# supply relative path from working directory (wd) to param-file
	-p your/path/to/input/fiducial_<2,3>zbins.param \
	# supply relative path from wd to output folder
	-o your/path/to/output/fiducial_<2,3>zbins/ \
	# supply relative path from wd to correctly set config-file (otherwise default.conf from MontePython will be used)
	--conf your/path/to/your_config.conf \
	# choose the MultiNest sampler (nested sampling)
	-m NS \
	# set an arbitrary but large number of steps (run should converge well before!)
	--NS_max_iter 10000000 \
	# do not use importance nested sampling
	--NS_importance_nested_sampling False \
	# for parameter estimation use 0.8 (0.3 recommended for more accurate evidences)
	--NS_sampling_efficiency 0.8 \
	# the more live points the smoother the contours, empirical number, experiment (depends also on hardware available)
	--NS_n_live_points 1000 \
	# run will finish/is converged if ln(Z_i) - ln(Z_j) <= NS_evidence_tolerance for i>j (0.5 is fine for parameters, for evidences you might want to lower it)  
	--NS_evidence_tolerance 0.5
