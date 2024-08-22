#!/usr/bin/env sh

# rm ./jax_spectral_dns/*.pyi
# stubgen jax_spectral_dns -o .
pip install --break-system-packages .
mypy ./tests/test_project.py
mypy jax_spectral_dns
