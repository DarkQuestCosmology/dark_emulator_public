# Dark Emulator
[![Anaconda-Server Badge](https://anaconda.org/nishimichi/dark_emulator/badges/version.svg)](https://anaconda.org/nishimichi/dark_emulator)
[![Anaconda-Server Badge](https://anaconda.org/nishimichi/dark_emulator/badges/latest_release_date.svg)](https://anaconda.org/nishimichi/dark_emulator)
[![Anaconda-Server Badge](https://anaconda.org/nishimichi/dark_emulator/badges/license.svg)](https://anaconda.org/nishimichi/dark_emulator)
[![Anaconda-Server Badge](https://anaconda.org/nishimichi/dark_emulator/badges/downloads.svg)](https://anaconda.org/nishimichi/dark_emulator)

A repository for emulators of our simulation suite.
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

You can then check if Dark Emulator works by running a tutorial notebook at
```
docs/tutorial.ipynb
docs/tutorial-hod.ipynb
```

