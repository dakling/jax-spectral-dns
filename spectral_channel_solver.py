#!/usr/bin/env python3

import jax.numpy as jnp
import jax.scipy as jsc
import matplotlib.pyplot as plt

import numpy as np
import scipy
from numpy.polynomial.chebyshev import Chebyshev

def cheb(n):
    def ret(x):
        ch = [1, x]
        for _ in range(2, n+1):
            ch.append(2 * x * ch[-1] - ch[-2])
        return ch[n]
    return ret

def slowct(u):
    N = len(u)
    xs = jnp.array([jnp.cos(jnp.pi * i / (N-1)) for i in range(N)])
    M = N
    u_cheb = []
    for m in range(M):
        p_m = 1 if m == 0 else 2
        out = 0
        for n in range(N):
            out += u[n] * cheb(m)(xs[n])
        u_cheb.append(p_m/N * out)
    return jnp.array(u_cheb)

def islowct(u_cheb):
    N = len(u_cheb)
    xs = jnp.array([jnp.cos(jnp.pi * i / (N-1)) for i in range(N)])
    u = []
    for n in range(N):
        out = 0
        for m in range(N):
            out += u_cheb[m] * cheb(m)(xs[n])
        u.append(out)
    return u

def fct(u, x):
    # adapted from https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform
    # TODO check if this is valid
    N = len(u)
    uFlipped = jnp.flip(u)
    # uFlipped = np.flip(u)
    u_cheb = jnp.sqrt(2/N) * jsc.fft.dct(uFlipped, norm="ortho") * jnp.array([1/jnp.sqrt(2) if i == 0 else 1 for i in range(N)])
    # u_cheb = np.sqrt(2/N) * scipy.fft.dct(uFlipped, norm="ortho") * np.array([1/jnp.sqrt(2) if i == 0 else 1.0 for i in range(N)])
    return jnp.array(u_cheb)

def ifct(u_cheb, k):
    # adapted from https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform
    N = len(u_cheb)
    u = jsc.fft.idct(jnp.sqrt(N/2) * u_cheb * jnp.array([jnp.sqrt(2) if i == 0 else 1 for i in range(N)]), norm="ortho")
    return u

def diffCheb(u, x, order=1):
# D = zeros(n);
# for i = 1:n
#     if mod(i,2) == 1
#         D(2*(1:((i-1)/2)),i) = 2*(i-1);
#     else
#         D(1,i) = i-1;
#         D(2*(2:((i)/2))-1,i) = 2*(i-1);
#     end
# end
    N = len(u)
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
    # return jnp.fft.ifft(jnp.fft.fft(u) * jnp.fft.fft(x) ** order)
    return u * k ** order

def eqF(u, x):
    uF = jnp.fft.fft(u)
    k = jnp.fft.fft(x)
    eq = diffFourier(uF, k, 2)
    return jnp.fft.ifft(eq)

def eqCheb(u, x):
    uCheb = fct(u, x)
    k = fct(x, x)
    eqCheb = diffCheb(uCheb, k, 2)
    eq = ifct(eqCheb, k)
    return eq

def perform_timestep(u, x, dt):
    print(eqCheb(u, x) * dt)
    return u + eqCheb(u, x) * dt
    # return u + eqF(u, x) * dt

def perform_simulation(u0, x, dt, number_of_steps):
    us = [u0]
    for _ in range(number_of_steps):
        us.append(perform_timestep(us[-1], x, dt))
    return us

def main():
    # xs = jnp.linspace(-1.0, 1.0, 50)
    # xs = np.linspace(-1.0, 1.0, 20)
    # xs = jnp.linspace(-1.0, 1.0, 50)
    N = 50
    # xs = jnp.array([np.cos(np.pi * i / N) for i in range(N+1)]) # gauss-lobatto points (SH2001, p. 488)
    xs = jnp.array([- np.cos(np.pi / N * (i + 1/2)) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)
    u = jnp.array([1 - x**2 for x in xs])
    # u = jnp.array([jnp.cos(x * jnp.pi) for x in xs])
    # u = jnp.array([1 for x in xs])
    # u = np.array([x for x in xs])
    # u = jnp.array([x for x in xs])
    us = perform_simulation(u, xs, 5e-5, 10)
    # ucheb = fct(u, xs)
    # xscheb = fct(xs, xs)
    # ducheb = diffCheb(ucheb, xscheb)
    # dducheb = diffCheb(ucheb, xscheb, 2)
    # dducheb = diffCheb(diffCheb(ucheb, xscheb), xscheb)
    # du = ifct(ducheb, xscheb)
    # ddu = ifct(dducheb, xscheb)
    # uu = ifct(ucheb, xscheb)
    # print(ucheb)
    # print(ducheb)
    # print(dducheb)
    # print(ddu)
    # print(du)
    # print(uu)
    # npcheb = Chebyshev(ucheb)
    # npchebsmpl = [npcheb(x) for x in xs]
    fig, ax = plt.subplots(1,1)
    # ax.plot(xs, u)
    # ax.plot(xs, npchebsmpl)
    # ax.plot(xs, ddu)
    # ax.plot(xs, uu)
    for u in us:
        ax.plot(xs, u)
    fig.savefig("plot.pdf")
