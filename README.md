# locMEA

locMEA is a source localization framework for Micro Electrode Array recordings.

## Getting Started

```
git clone https://github.com/uranc/locmea.git
```

Check the tutorials directory for different use cases and tutorials.

### Prerequisities

locMEA is developed in python. MATLAB extensions will eventually be.
Minimum packages required are:

There is 4 main modules of the framework with different functions.

# getData
Load the data, visualize, pre-process
 - Requierements - 
numpy 
h5py (Optional for hdf5 files)
Pickle (Optional for pickle files)


# setInverseProblem
Define the inverse source localization problem using the recordings and the electrode geometry.

numpy
scipy

# setOptimizationProblem

numpy
CasADi
OpenMP (optional)

CasADi is an open-source symbolic framework for algorithmic (a.k.a. automatic) differentiation and numeric optimization.  requires more packages, check the website for installation details.

```
https://github.com/casadi/casadi/wiki
```

# getVisualization
Matplotlib
For optimization problems: CasADi

### Installing

Currently the project is in developement, you can check the repo to have an idea.


```
git clone https://github.com/uranc/locmea.git
```

Check the tutorials directory for different use cases and tutorials.
PyPi package coming soon..

## Running the tests

*Not yet* 
Tests are not yet implemented.. 


## Deployment

Using the CasADi framework, you can generate the C-code for developed algorithms which will be much faster and will allow hardware implementations.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

*Not yet* 

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

locMEA is an open-source tool, written in Python and built on top of the frameworks mentioned as earlier. It is developed by **Cem Uran** - [uranc](https://github.com/uranc) of the University of Freiburg under the supervision of **Prof. Stefan Rotter**  - [Bernstein Center Freiburg](https://www.bcf.uni-freiburg.de/people/details/rotter)

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3, meaning the code can be used royalty-free even in commercial applications. See the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Thanks to all the great developers that are used in this framework. 
- CasADi, LFPy, NEURON
- numpy, scipy, matplotlib, h5py, pickle, Doxygen

* Thanks to BWUniCluster for allowing the resources and the all computational power.

