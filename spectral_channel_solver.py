#!/usr/bin/env python3

import jax.numpy as jnp
import jax.scipy as jsc
import matplotlib.pyplot as plt

def cheb(n):
    def ret(x):
        ch = [1, x]
        for _ in range(2, n+1):
            ch.append(2 * x * ch[-1] - ch[-2])
        return ch[n]
    return ret

def fct(u, x):
    # adapted from https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform
    # TODO use a different basis to enforce BCs (or manipulate matrix?)
    N = len(u)
    uFlipped = jnp.flip(u)
    u_cheb = jnp.sqrt(2/N) * jsc.fft.dct(uFlipped, norm="ortho") * jnp.array([1/jnp.sqrt(2) if i == 0 else 1 for i in range(N)])
    return jnp.array(u_cheb)

def ifct(u_cheb, k):
    # adapted from https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform
    N = len(u_cheb)
    u = jsc.fft.idct(jnp.sqrt(N/2) * u_cheb * jnp.array([jnp.sqrt(2) if i == 0 else 1 for i in range(N)]), norm="ortho")
    return u

def diffCheb(u, x, order=1):
    N = len(u)
    # TODO save matrix for better performance at some point
    ret = [[0.0 for j in range(N)] for i in range(N)]
    for i in range(N):
        if i % 2 == 0: # even
            for j in range(0, i, 2):
                ret[j + 1][i] = 2.0 * i
        else: # odd
            ret[0][i] = i
            for j in range(2, i, 2):
                ret[j][i] = 2.0 * i
    retArr = jnp.array(ret)
    return jnp.linalg.matrix_power(retArr, order) @ u


def diffFourier(u, k, order=1):
    return u * k ** order

def eqF(u, x):
    uF = jnp.fft.fft(u)
    k = jnp.fft.fft(x)
    eq = diffFourier(uF, k, 2)
    return jnp.fft.ifft(eq)

def eqCheb(u, x):
    N = len(u)
    uCheb = fct(u, x)
    k = fct(x, x)
    eqCheb = diffCheb(uCheb, k, 2)
    eq = ifct(eqCheb, k)
    return eq

def perform_timestep(u, x, dt):
    return u + eqCheb(u, x) * dt
    # return u + eqF(u, x) * dt

def perform_simulation(u0, x, dt, number_of_steps):
    us = [u0]
    for _ in range(number_of_steps):
        us.append(perform_timestep(us[-1], x, dt))
    return us

def main():
    N = 50
    xs = jnp.array([- jnp.cos(jnp.pi / N * (i + 1/2)) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)
    u = jnp.array([1 - x**2 for x in xs])
    us = perform_simulation(u, xs, 5e-4, 3)
    fig, ax = plt.subplots(1,1)
    for u in us:
        ax.plot(xs, u)
    fig.savefig("plot.pdf")
