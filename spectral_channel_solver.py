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

def jct_coeff_0(u):
    N = len(u)
    xs = jnp.array([- jnp.cos(jnp.pi / N * (i + 1/2)) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)
    out = 0.0
    for i in range(N):
        out += u[i] * (2 * cheb(i+2)(xs[i]) - cheb(i)(xs[i]))
        # out += u_cheb[i] * (cheb(i)(x))
    p = 1
    # return  p * jnp.sqrt(2/N) * out
    # return  p / N * out
    return  out

def dct_coeff(u, m):
    N = len(u)
    # xs = jnp.array([- jnp.cos(jnp.pi / N * (i + 1/2)) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)
    out = 0
    for n in range(N):
        # out += p_m * u[n] * cheb(m)(xs[n])
        # out += p_m * u[n] * cheb(m)(xs[n])
        out += u[n] * jnp.cos(jnp.pi/N * (n+1/2) * m)
    return out * jnp.sqrt(2/N)

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
        T_raw = jnp.vstack(jnp.array([[ cheb(m)(xs[n]) for m in range(N+2)] for n in range(N)]))
        E = jnp.identity(N+4)
        M = (jnp.stack([(E[:,2:] - E[:,:-2]).transpose() for _ in range(N)]))[:, 0:, 0:-2]
        M_formatted = M[:, :-2, :]
        T = (M_formatted @ T_raw.reshape(N, N+2, 1)).reshape(N, N)
        Tinv = jnp.linalg.inv(T)
    return Tinv @ u

def isct(u_cheb, T=None):
    if type(T) == NoneType:
        N = len(u_cheb)
        xs = jnp.array([- jnp.cos(jnp.pi / N * (i + 1/2)) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)
        T_raw = jnp.vstack(jnp.array([[ cheb(m)(xs[n]) for m in range(N+2)] for n in range(N)]))
        T = T_raw[:,2:] - T_raw[:,:-2]
        # N = len(u_cheb)
        # def ret(x):
        #     out = 0
        #     for i in range(N):
        #         out += u_cheb[i] * (cheb(i+2)(x) - cheb(i)(x))
        #         # out += u_cheb[i] * (cheb(i)(x))
        #     return out
        # xs = jnp.array([- jnp.cos(jnp.pi / N * (i + 1/2)) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)
        # u = jnp.array([ret(xs[i]) for i in range(N)])
    return T @ u_cheb

    # xs = jnp.array([- jnp.cos(jnp.pi / N * (i + 1/2)) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)
    # T = jnp.array([[ cheb(m)(xs[n]) for n in range(N)] for m in range(N)])
    # return T @ u_cheb

def assembleChebDiffMat(N, order=1):
    matL = [[0.0 for j in range(N+2)] for i in range(N+2)]
    for i in range(N+2):
        if i % 2 == 0: # even
            for j in range(0, i, 2):
                if j+1 < len(matL):
                    matL[j + 1][i] = 2.0 * i
        else: # odd
            matL[0][i] = i
            for j in range(2, i, 2):
                matL[j][i] = 2.0 * i
    E = jnp.identity(N+4)
    # M = (jnp.stack([(E[:,2:] - E[:,:-2]).transpose() for _ in range(N)]))[:, 0:, 0:-2]
    M = (E[:,2:] - E[:,:-2]).transpose()[:, :-2]
    # M_formatted = M[:, :-2, :]
    Minv = jnp.linalg.inv(M)
    mat_raw = jnp.vstack(jnp.array(matL))
    # mat = mat_raw[:, 2:] - mat_raw[:, :-2]
    mat = jnp.linalg.matrix_power(mat_raw, order) @ Minv
    # return jnp.linalg.matrix_power(mat, order)
    return jnp.stack([mat for _ in range(N)])

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
    uCheb = sct(u, Tinv)
    eqCheb = diffCheb(uCheb, 2, mat)
    # eq = ifct(eqCheb)
    eq = isct(eqCheb, T)
    return eq

def perform_timestep(u, x, dt, T=None, Tinv=None, mat=None):
    _ = x
    return u + eqCheb(u, T, Tinv, mat) * dt
    # return u + eqF(u, x) * dt

def perform_simulation(u0, x, dt, number_of_steps):
    N = len(u0)
    xs = x
    T_raw = jnp.vstack(jnp.array([[ cheb(m)(xs[n]) for m in range(N+2)] for n in range(N)], dtype=float))
    E = jnp.eye((N, N, N+2))
    M = E[:,:,2:] - E[:,:,:-2]
    T = E @ T_raw
    T_alt = T_raw[:,2:] - T_raw[:,:-2]
    Tinv = jnp.linalg.inv(T)
    mat = assembleChebDiffMat(N, 1)
    print("Done with matrix assembly")
    us = [u0]
    for _ in range(number_of_steps):
        us.append(perform_timestep(us[-1], x, dt, T, Tinv, mat))
    return us

def main():
    N = 5
    # xs = jnp.array([- jnp.cos(jnp.pi / N * (i + 1/2)) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)
    xs = jnp.array([- jnp.cos(jnp.pi / N * (i + 1/2)) for i in range(N)]) # gauss-lobatto points (SH2001, p. 488)
    # xs = jnp.cos(jnp.pi*jnp.linspace(N,0,N+1)/N) # gauss-lobatto points with boundaries
    # print(xs)
    # u = jnp.array([1 - x**2 for x in xs])
    # u = jnp.array([4 * x for x in xs])
    # u = jnp.array([jnp.cos(x * jnp.pi) for x in xs])
    u = jnp.array([2*x**2 - 2 for x in xs])
    # u = jnp.array([1-x**2 for x in xs])
    # u = jnp.array([1 for x in xs])
    print(u)
    # uCheb = fct(u)
    uCheb = sct(u)
    # uCheb = jnp.array([1.0 if i == 0 else 0.0 for i in range(N)])
    print(uCheb)
    # uu = ifct(uCheb)
    duCheb = diffCheb(uCheb)
    print(duCheb)
    uu = isct(uCheb)
    du = isct(duCheb)
    print(du)
    # us = perform_simulation(u, xs, 5e-5, 40)
    # print(us[-1])
    fig, ax = plt.subplots(1,1)
    # for u in us:
    #     ax.plot(xs, u)
    ax.plot(xs, u)
    ax.plot(xs, uu)
    ax.plot(xs, du)

    fig.savefig("plot.pdf")
