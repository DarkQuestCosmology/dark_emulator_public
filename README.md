# Dark Emulator
[![Anaconda-Server Badge](https://anaconda.org/nishimichi/dark_emulator/badges/version.svg)](https://anaconda.org/nishimichi/dark_emulator)
[![Anaconda-Server Badge](https://anaconda.org/nishimichi/dark_emulator/badges/latest_release_date.svg)](https://anaconda.org/nishimichi/dark_emulator)
[![Anaconda-Server Badge](https://anaconda.org/nishimichi/dark_emulator/badges/license.svg)](https://anaconda.org/nishimichi/dark_emulator)
[![Anaconda-Server Badge](https://anaconda.org/nishimichi/dark_emulator/badges/downloads.svg)](https://anaconda.org/nishimichi/dark_emulator)

A repository for a cosmology tool `dark_emulator` to emulate halo clustering statistics. The code is developed based on Dark Quest simulation suite (https://darkquestcosmology.github.io/). The current version supports the halo mass function and two point correlation function (both halo-halo and halo-matter cross).

## Install
In order to install dark emulator package, use pip:
```
   pip install dark_emulator
```
or use conda:
```
   conda install -c nishimichi dark_emulator
```
If the above does not work for you, you may download the source files from this repository and install via
```
python -m pip install -e .
```
after moving to the top directory of the source tree.
In that case, you need to install `pyfftlog`, `george` (a software package for the Gaussian process) and colossus
```
conda install -c conda-forge george
conda install -c conda-forge pyfftlog
pip install colossus
```

## Usage
You can then check how Dark Emulator works by running a tutorial notebook at
```
docs/tutorial.ipynb
docs/tutorial-hod.ipynb
```
See also the documentation on [readthedocs](https://dark-emulator.readthedocs.io/en/latest/).

## Code Paper
The main reference for our halo emulation strategy is: "Dark Quest. I. Fast and Accurate Emulation of Halo Clustering Statistics and Its Application to Galaxy Clustering", by T. Nishimichi et al., [ApJ 884, 29 (2019)](https://iopscience.iop.org/article/10.3847/1538-4357/ab3719/meta), [arXiv:1811.09504](https://arxiv.org/abs/1811.09504). Please also refer to the paper "Cosmological inference from emulator based halo model I: Validation tests with HSC and SDSS mock catalogs", by H. Miyatake et al.,  [arXiv:2101.00113](https://arxiv.org/abs/2101.00113) for the implementation and performance of the halo-galaxy connection routines.

