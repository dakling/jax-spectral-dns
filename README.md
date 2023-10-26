# Jax-Optim

This repository contains a spectral solver written in python. Notably, the
library [jax](https://github.com/google/jax) is used, so that automatic
differentiation is supported. The main purpose of this project is to implement a
Navier-Stokes DNS solver. 

## Getting started

### Dependencies

- python, obviously
- some standard python libraries (mostly used for post-processing):
    - numpy
    - scipy
    - matplotlib
- [jax](https://github.com/google/jax)
- (notably, [jax-cfd](https://github.com/google/jax-cfd) is not needed.)

### Running a case

Check out the functions defined in `test.py` for examples. The functions
starting with "test" can also be used in order to check that everything is
working fine. Functions that run the solver but to not contain any quantitative
test start with "run".

## Example outputs

### run_jimenez_1990

Result of the function `run_jimenez_1990` in `test.py`, which reproduces the
case documented in Figure 1 of "Transition to turbulence in two-dimensional
Poiseuille flow" by Javier Jimenez (1990, Journal of Fluid Mechanics, vol 218,
pp 265-297)

![vorticity at Re 5000]( ./img/vort_z_Re_5000_jimenez1990.gif )

![vorticity at Re 5000 (isolines)]( ./img/vort_z_Re_5000_isolines_jimenez1990.gif )
