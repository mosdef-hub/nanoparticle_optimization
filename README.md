# NanoOpt: Deriving potentials for coarse-grained nanoparticles via potential-matching

# Installation

NanoOpt is centered around the [mBuild package](https://github.com/mosdef-hub/mbuild) which is easiest to install using the Anaconda package manager. The recommended installation instructions are as follows:

#### NOTE

For the version of mbuild used in this paper and study, use `environment.yml`.
For a newer version of the packages with improved mBuild features, use `environment-dev.yml`.

1. Clone NanoOpt:
```sh
git clone https://github.com/summeraz/nanoparticle_optimization
cd nanoparticle_optimization
```

2. Determine what version you want to use:
Refer to the note above

* If you want to emulate the study and paper:
```
git checkout bdaa8b8
conda env create -f environment.yml
conda activate nanoopt-binder
pip install .
```

* If you want to use newer package versions
```
conda env create -f environment.yml
conda activate nanoopt-binder
pip install .
```



# Tutorials

The `tutorials` directory contains several tutorials in the form of Jupyter notebooks that provide examples of the optimization code.
These can be accessed online via Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mosdef-hub/nanoparticle_optimization/master)
However, these are best run locally as several of the code blocks can take a considerable amount of time to run on Binder.
