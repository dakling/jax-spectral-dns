#!/usr/bin/env python3

from types import NoneType
import jax.numpy as jnp
import jax.scipy as jsc
from matplotlib import legend
import matplotlib.pyplot as plt
from matplotlib import legend
from numpy import float128
import scipy as sc

def get_cheb_grid(N):
    # return jnp.array([jnp.cos(jnp.pi / (N) * i) for i in range(N+1)]) # gauss-lobatto points with endpoints
    return jnp.array([jnp.cos(jnp.pi / (N-1) * i) for i in range(N)]) # gauss-lobatto points with endpoints

def get_fourier_grid(N):
    if N % 2 != 0:
        print("Warning: Only even number of points supported for Fourier basis, making the domain larger by one.")
        N += 1
    return jnp.linspace(0.0, 2*jnp.pi, N+1)[1:]

def cheb(n):
    def ret(x):
        ch = [1, x]
        for _ in range(2, n+1):
            ch.append(2 * x * ch[-1] - ch[-2])
        return ch[n]
    return ret

def pad_mat_with_zeros(mat):
    return jnp.block([[jnp.zeros((1, mat.shape[1]+2))], [jnp.zeros((mat.shape[0], 1)), mat, jnp.zeros((mat.shape[0], 1))], [jnp.zeros((1, mat.shape[1]+2))]])

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

def assembleChebDiffMat(N_, order=1):
    xs = get_cheb_grid(N_)
    N_ = len(xs)
    c = jnp.block([2.0, jnp.ones((1, N_-2))*1.0, 2.0]) * (-1)**jnp.arange(0,N_)
    X = jnp.repeat(xs, N_).reshape(N_, N_)
    dX = X - jnp.transpose(X)
    D_ = jnp.transpose((jnp.transpose(1/c) @ c)) / (dX + jnp.eye(N_))
    return D_ - jnp.diag(sum(jnp.transpose(D_)))

def assembleFourierDiffMat(N, order=1):
    if N % 2 != 0:
        raise Exception("Fourier discretization points must be even!")
    h = 2*jnp.pi/N;
    column = jnp.block([0, .5*(-1)**jnp.arange(1,N) * 1/jnp.tan(jnp.arange(1,N) * h/2)])
    column2 = jnp.block([column[0], jnp.flip(column[1:])])
    return jnp.linalg.matrix_power(jsc.linalg.toeplitz(column, column2), order)

def diffCheb(u, order=1, mat=None):
    N = len(u)
    if type(mat) == NoneType:
        mat = assembleChebDiffMat(N, order)
    return mat @ u

def diffFourier(u, order=1, mat=None):
    if type(mat) == NoneType:
        mat = assembleFourierDiffMat(len(u), order)
    return mat @ u

def eq_cheb(u, mat=None):
    return diffCheb(u, 2, mat=mat)

def perform_timestep_cheb_1D (u, dt, mat=None):
    new_u = u + eq_cheb(u, mat) * dt
    new_u_bc = jnp.block([0.0, new_u[1:-1], 0.0])
    return new_u_bc
    # return u + eqF(u, x) * dt

def perform_simulation_cheb_1D(u0, dt, number_of_steps):
    N = len(u0)
    cheb_mat_1 = assembleChebDiffMat(N, 1)
    cheb_mat_2 = jnp.linalg.matrix_power(cheb_mat_1, 2)
    # fourier_mat_1 = assembleChebDiffMat(N, 1)
    # fourier_mat_2 = jnp.linalg.matrix_power(fourier_mat_1, 2)
    print("Done with matrix assembly")
    us = [u0]
    for _ in range(number_of_steps):
        us.append(perform_timestep_cheb_1D (us[-1], dt, cheb_mat_2))
    return us

def run_cheb_sim_1D():
    Ny = 24
    ys = get_cheb_grid(Ny)
    # u = jnp.array([1 - x**2 for x in ys])
    u = jnp.array([jnp.cos(x * jnp.pi/2) for x in ys])

    steps = 1000
    dt = 5e-4

    final_time = steps*dt
    us = perform_simulation_cheb_1D(u, dt, steps)
    u_ana = list(map(lambda x: jnp.exp(-jnp.pi**2 * final_time/4) * jnp.cos(x * jnp.pi/2), ys))
    print("maximum of final profile: " + str(max(us[-1])))
    print("analytical: " + str(max(u_ana)))
    fig, ax = plt.subplots(1,1)
    for u in us:
        ax.plot(ys, u)
    ax.plot(ys, u_ana)
    fig.savefig("plot.pdf")

def eq_fourier(u, mat=None):
    return diffFourier(u, 2, mat=mat)

def perform_timestep_fourier_1D (u, dt, mat=None):
    new_u = u + eq_fourier(u, mat) * dt
    return new_u
    # return u + eqF(u, x) * dt

def perform_simulation_fourier_1D(u0, dt, number_of_steps):
    N = len(u0)
    mat_1 = assembleFourierDiffMat(N, 1)
    mat_2 = jnp.linalg.matrix_power(mat_1, 2)
    print("Done with matrix assembly")
    us = [u0]
    for _ in range(number_of_steps):
        us.append(perform_timestep_fourier_1D (us[-1], dt, mat_2))
    return us

def run_fourier_sim_1D():
    Nx = 80
    xs = get_fourier_grid(Nx)
    u = jnp.array([jnp.cos(x) for x in xs])

    steps = 10000
    dt = 5e-6

    final_time = steps*dt
    us = perform_simulation_fourier_1D(u, dt, steps)
    # u_ana = list(map(lambda x: jnp.exp(-jnp.pi**2 * final_time/4) * jnp.cos(x), xs))
    print("maximum of final profile: " + str(max(us[-1])))
    # print("analytical: " + str(max(u_ana)))
    fig, ax = plt.subplots(1,1)
    for u in us:
        ax.plot(xs, u)
    # ax.plot(xs, u_ana)
    fig.savefig("plot.pdf")

def eq_cheb_2D(u, mat=None):
    return mat@u
    # return lap_u

def perform_timestep_cheb_2D (u, dt, mat):
    new_u = u + eq_cheb_2D(u, mat) * dt
    N = int(jnp.sqrt(len(new_u)))
    new_u_bc = (pad_mat_with_zeros(new_u.reshape((N, N))[1:-1,1:-1])).flatten()
    return new_u_bc

def perform_simulation_cheb_2D(u0, dt, number_of_steps):
    Nx = int(jnp.sqrt(u0.shape[0]))
    Ny = Nx
    cheb_mat_1 = assembleChebDiffMat(Nx, 1)
    cheb_mat_2 = jnp.linalg.matrix_power(cheb_mat_1, 2)
    id_x = jnp.eye(Nx)
    id_y = jnp.eye(Ny)
    cheb_mat_xx = jnp.kron(id_x, cheb_mat_2)
    cheb_mat_yy = jnp.kron(cheb_mat_2, id_y)
    lap_mat = cheb_mat_xx + cheb_mat_yy
    print("Done with matrix assembly")
    us = [u0]
    for _ in range(number_of_steps):
        us.append(perform_timestep_cheb_2D (us[-1], dt, lap_mat))
    return us

def run_cheb_cheb_sim_2D():
    Nx = 24
    Ny = Nx
    xs = get_cheb_grid(Nx)
    # ys = get_cheb_grid(Ny)
    ys = xs

    XX_, YY_ = jnp.meshgrid(xs, ys)
    XX = jnp.transpose(XX_).flatten()
    YY = jnp.transpose(YY_).flatten()

    # u = jnp.array([1 - x**2 for x in ys])
    u = jnp.array([(1 - y**8) * jnp.cos(x * jnp.pi/2) for (x,y) in zip(XX, YY)])
    # u = jnp.array([10*jnp.sin(8*x*(y-1)) for (x,y) in zip(XX, YY)])

    steps = 1000
    dt = 5e-5

    # final_time = steps*dt
    us = perform_simulation_cheb_2D(u, dt, steps)
    # u_ana = list(map(lambda x: jnp.exp(-jnp.pi**2 * final_time/4) * jnp.cos(x * jnp.pi/2), ys))
    # print("maximum of final profile: " + str(max(us[-1])))
    # print("analytical: " + str(max(u_ana)))
    fig, ax = plt.subplots(1,1)
    for u in us:
        ax.plot(ys, u.reshape((Nx,Ny))[Nx//2, :])
    # ax.plot(ys, u_ana)
    fig.savefig("plot_y.pdf")

    fig, ax = plt.subplots(1,1)
    for u in us:
        ax.plot(xs, u.reshape((Nx,Ny))[:, Ny//2])
    # ax.plot(ys, u_ana)
    fig.savefig("plot_x.pdf")

    fig, ax = plt.subplots(1,2, subplot_kw={"projection": "3d"})
    ax[0].plot_surface(XX_, YY_, us[0].reshape((Nx,Ny)).transpose())
    ax[1].plot_surface(XX_, YY_, us[-1].reshape((Nx,Ny)).transpose())
    # ax.plot(ys, u_ana)
    fig.savefig("plot3d.pdf")

def main():
    # run_cheb_sim_1D()
    # run_fourier_sim_1D()
    run_cheb_cheb_sim_2D()
