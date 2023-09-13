#!/usr/bin/env python3

from types import NoneType
import jax.numpy as jnp
import jax.scipy as jsc
import matplotlib.pyplot as plt
from numpy import float128

import scipy as sc

def cheb(n):
    def ret(x):
        ch = [1, x]
        for _ in range(2, n+1):
            ch.append(2 * x * ch[-1] - ch[-2])
        return ch[n]
    return ret

def get_xs(N):
    return jnp.array([jnp.cos(jnp.pi / N * (i + 1/2)) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)

def get_xs_with_bnd(N):
    return jnp.array([jnp.cos(jnp.pi / (N-1) * i) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)

def cheb_fft_diff(u):
    # adapted from Trefethen
    N = len(u)
    xs = get_xs_with_bnd(N)
    ii = jnp.arange(0, N, 1)
    II = jnp.block([ii[:-1], 0, -jnp.flip(ii[1:-1])])
    V = jnp.block([u, jnp.flip(u[1:-1])])
    U = jnp.sqrt(len(V)) * jnp.fft.fft(V, norm="ortho").real
    W = jnp.fft.ifft((0+1j * II) * U / jnp.sqrt(len(V)), norm="ortho").real
    w_0 = jnp.sum(ii**2 * U[ii] / (N-1) + 0.5 * N * U[N-1])
    w_end = jnp.sum((-1)**(ii+1.0) * ii**2 * U[ii] / (N-1) + 0.5 * (-1)**(N) * (N - 1) * U[N-1])
    w = jnp.block([w_0, -W[1:N-1] / jnp.array([jnp.sqrt(1-x**2) for x in xs[1:N-1]]), w_end])
    return w

def fct(u):
    # adapted from https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform
    # TODO use a different basis to enforce BCs (or manipulate matrix?)
    N = len(u)
    uFlipped = jnp.flip(u)
    # u_cheb = jnp.sqrt(2/N) * jsc.fft.dct(uFlipped, norm="ortho") * jnp.array([1/jnp.sqrt(2) if i == 0 else 1 for i in range(N)])
    u_dct = jsc.fft.dct(uFlipped, norm="ortho")
    u_dct_ext = jnp.block([u_dct, dct_coeff(uFlipped, N), dct_coeff(uFlipped, N+1)])
    u_cheb_0 = jct_coeff_0(u)
    # u_cheb_bc = jnp.sqrt(2/N) * jnp.block([u_cheb_0, (u_dct_ext[2:] - u_dct_ext[:-2])[1:]]) * jnp.array([1/jnp.sqrt(2) if i == 0 else 1 for i in range(N)])
    u_cheb_bc = jnp.block([u_cheb_0, (u_dct_ext[2:] - u_dct_ext[:-2])[1:]]) * jnp.array([1/jnp.sqrt(2) if i == 0 else 1 for i in range(N)])
    return u_cheb_bc

def ifct(u_cheb):
    # adapted from https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform
    N = len(u_cheb)
    u = jsc.fft.idct(jnp.sqrt(N/2) * u_cheb * jnp.array([jnp.sqrt(2) if i == 0 else 1 for i in range(N)]), norm="ortho", n = N)
    # u_bc = jnp.block([0.0, u[2:] - u[:N-2], 0.0])
    u_bc = u[2:] - u[:N-2]
    # u_bc = u[2:] - u[:N-2]
    return u_bc

def sct(u, Tinv=None):
    if type(Tinv) == NoneType:
        N = len(u)
        xs = jnp.array([- jnp.cos(jnp.pi / N * (i + 1/2)) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)
        T = jnp.vstack(jnp.array([[ cheb(m)(xs[n]) for m in range(N)] for n in range(N)]))
        Tinv = jnp.linalg.inv(T)
    return Tinv @ u

def isct(u_cheb, T=None):
    if type(T) == NoneType:
        N = len(u_cheb)
        xs = jnp.array([- jnp.cos(jnp.pi / N * (i + 1/2)) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)
        T = jnp.vstack(jnp.array([[ cheb(m)(xs[n]) for m in range(N)] for n in range(N)]))
    return T @ u_cheb

    # xs = jnp.array([- jnp.cos(jnp.pi / N * (i + 1/2)) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)
    # T = jnp.array([[ cheb(m)(xs[n]) for n in range(N)] for m in range(N)])
    # return T @ u_cheb

def assembleChebDiffMat(N, order=1):
    matL = [[0.0 for j in range(N)] for i in range(N)]
    for i in range(N):
        if i % 2 == 0: # even
            for j in range(0, i, 2):
                if j+1 < len(matL):
                    matL[j + 1][i] = 2.0 * i
        else: # odd
            matL[0][i] = i
            for j in range(2, i, 2):
                matL[j][i] = 2.0 * i
    mat = jnp.vstack(jnp.array(matL))
    return jnp.linalg.matrix_power(mat, order)

def diffCheb(u, order=1, mat=None):
    N = len(u)
    if type(mat) == NoneType:
        mat = assembleChebDiffMat(N, order)
    u_ext = jnp.block([u, 0, 0]) # TODO
    return jnp.linalg.matrix_power(mat, order) @ u_ext


def diffFourier(u, k, order=1):
    return u * k ** order

def eqF(u, x):
    uF = jnp.fft.fft(u)
    k = jnp.fft.fft(x)
    eq = diffFourier(uF, k, 2)
    return jnp.fft.ifft(eq)

def eqCheb(u, T=None, Tinv=None, mat=None):
    # uCheb = fct(u)
    # uCheb = sct(u, Tinv)
    eq = cheb_fft_diff(cheb_fft_diff(u))
    # eq = ifct(eqCheb)
    # eq = isct(eqCheb, T)
    return jnp.block([0.0, eq[1:-1], 0.0])

def perform_timestep(u, x, dt, T=None, Tinv=None, mat=None):
    _ = x
    return u + eqCheb(u, T, Tinv, mat) * dt
    # return u + eqF(u, x) * dt

def perform_simulation(u0, x, dt, number_of_steps):
    us = [u0]
    for _ in range(number_of_steps):
        us.append(perform_timestep(us[-1], x, dt))
    return us

def main():
    N = 10
    # xs = jnp.array([- jnp.cos(jnp.pi / N * (i + 1/2)) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)
    # xs = jnp.array([- jnp.cos(jnp.pi / N * (i + 1/2)) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)
    xs = get_xs_with_bnd(N)
    # xs = jnp.cos(jnp.pi*jnp.linspace(N,0,N+1)/N) # gauss-lobatto points with boundaries
    # print(xs)
    # u = jnp.array([1 - x**2 for x in xs])
    # u = jnp.array([4 * x for x in xs])
    # u = jnp.array([jnp.cos(x * jnp.pi) for x in xs])
    # u = jnp.array([2*x**2 - 2 for x in xs])
    u = jnp.array([1-x**2 for x in xs])
    # u = jnp.array([x for x in xs])
    # du = cheb_fft_diff(u)
    # ddu = cheb_fft_diff(du)
    # print(u)
    # print(du)
    # print(ddu)
    # uCheb = fct(u)
    # uCheb = sct(u)
    # uCheb = jnp.array([1.0 if i == 0 else 0.0 for i in range(N)])
    # print(uCheb)
    # uu = ifct(uCheb)
    # duCheb = diffCheb(uCheb)
    # duCheb = cheb_fft_diff(uCheb, 1)
    # uu = isct(uCheb)
    # du = isct(duCheb)
    # print(du)
    us = perform_simulation(u, xs, 5e-5, 500)
    print(max(us[0]))
    print(us[0])
    print(max(us[-1]))
    print(us[-1])
    fig, ax = plt.subplots(1,1)
    for u in us:
        ax.plot(xs, u)
    # ax.plot(xs, u)
    # ax.plot(xs, uu)
    # ax.plot(xs, du)

    fig.savefig("plot.pdf")
