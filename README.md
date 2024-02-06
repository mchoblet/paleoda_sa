# Multi-time scale Paleoclimate Data Assimilation for the reconstruction of South American Hydroclimate

This repository contains all the code needed to reproduce the results and figures of publication A continental reconstruction of hydroclimatic variability in South America
during the past 2000 years, by Mathurin Choblet et al., to be submitted to Climate of the Past (February 2024).

It contains the code to reconstruct *past climates* using *climate model data* and *climate proxy data* using the Paleoclimate Data Assimilation method. This repository in particular implements an efficient multi-time scale method to allow for using climate proxy data that is not dated annually. While the reconstruction here is performed specifically for the South American continent, it can also be equally applied for global climate field reconstructions.

The code is a documented and reduced version of an older repository, which I used for my masters thesis (https://github.com/mchoblet/paleoda). The pure Ensemble Kalman Filter algorithm code underlying the algorithm is also stored in https://github.com/mchoblet/ensemblefilters. 

For questions and comments feel free to contact me: mathurin [AT] choblet.com
I’m happy to give any kind of insight into the code.

# Content
* ‘paper_experiments.ipynb’ contains the configurations of the reconstruction experiments and starts the algorithm by calling functions in the ‘algorithm’ folder
* ‘Paper_plots.ipynb’ contains all plotting code to reproduce the figures
* The ‘algorithm’ folder contains everything related to the reconstruction algorithm in three python scripts (wrapper_new.py which runs the reconstruction, using functions from utils_new.py and the kalman filter algorithms from kalman_filters.py)
* The ‘data_preprocessing’ contains jupyter-notebook in which the climate model data and proxy record data is standardized and prepared for its use in the paper_experiments notebook. It also contains code for creating speleothem composites, creating the proxy record files and performing linear regressions of climate archives (tree rings and corals) to instrumental data. The notebooks in this folder are mainly stored for documentary reasons if a user would like to know exactly what we did or would like to use the code for different input data. It is not necessary to run these notebooks to run the reconstructions (see next section.)


All scripts and notebooks contain a fair amount of description and comments.

# Data needed to run the code:
Climate model data and proxy record data to run the reconstructions is made available via a Zenodo repository: https://zenodo.org/records/10370001. Note, that this is the data after preprocessing it via the jupyter notebooks in the ‘data_preprocessing’ folder. 
The results of the reconstructions, which can be used to reproduce the figures of the publications are published in another Zenodo repository: https://zenodo.org/records/10622265

# Run the code
To run the reconstructions, follow the steps in the paper_experiments.ipynb notebook. Change filepaths in the config dictionary to where the input data is stored and where the output data shall be stored.

# Dependencies:
Running the climate field reconstructions primarily relies on the following packages: 
* Xarray
* Numpy
* Scipy
* Cftime for decoding time units in netcdf files
* Matplotlib
* Tqdm for a progress bar

You can install the conda environment with which the reconstructions were run.
    ``` 
    conda env create -f paleoda_environment.yml
    ``` 

# Recommend computational resources for running reconstructions.
At least 16gb of RAM due to input data file sizes. The more cpu cores you have the better. The math behind the reconstruction is implemented via Numpy Array operations, which automatically makes use of all available cores.
I recommend testing the reconstruction time for a single Monte Carlo iteration or year before running it for the entire reconstruction. 

# Cite this work:

# Acknowledgement
When I started this project in 2022, I was heavily inspired by the code of the Last Millennium Reanalysis (LMR, https://github.com/modons/LMR, Hakim et al https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016jd024751), which is not up to date (e.g. no xarray support, slower kalman filter implementation ...). Even if the code here is finally very different from the LMR code, this work would not have been possible without LMR.
