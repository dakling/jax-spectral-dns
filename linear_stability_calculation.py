#!/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
from numpy.linalg import svd
from scipy.linalg import eig, cholesky
from scipy.sparse.linalg import eigs
from scipy.integrate import quad
import scipy
import matplotlib.pyplot as plt
import timeit

from importlib import reload
import sys

from navier_stokes import NavierStokesVelVort

try:
    reload(sys.modules["cheb"])
except:
    pass
from cheb import cheb, phi

try:
    reload(sys.modules["domain"])
except:
    pass
from domain import Domain

try:
    reload(sys.modules["field"])
except:
    pass
from field import Field, VectorField

NoneType = type(None)


class LinearStabilityCalculation:
    def __init__(self, Re=180.0, alpha=3.25, n=50):
        self.Re = Re
        self.alpha = alpha
        # self.n = int(n * Domain.aliasing)  # chebychev resolution
        self.n = n  # chebychev resolution

        self.A = None
        self.B = None
        self.eigenvalues = None
        self.eigenvectors = None

        self.C = None
        self.growth = []

        # self.ys = [np.cos(np.pi * (2*(i+1)-1) / (2*self.n)) for i in range(self.n)] # gauss-lobatto points (SH2001, p. 488)
        domain = Domain((n,), (False,))
        self.ys = domain.grid[0]

        self.velocity_field_ = None

    def assemble_matrix_fast(self):
        alpha = self.alpha
        Re = self.Re

        ys = self.ys
        n = len(ys)

        noOfEqs = 4
        N = n * noOfEqs
        A = np.zeros([N, N], dtype=complex)
        B = np.zeros([N, N], dtype=complex)

        I = 0 + 1j

        def local_to_global_index(j, k, eq, var):
            jj = j + eq * n
            kk = k + var * n
            return (jj, kk)

        u_fun = lambda y: (1 - y**2)
        du_fun = lambda y: -2 * y
        beta = 0
        kSq = alpha**2 + beta**2
        for j in range(n):
            y = ys[j]
            U = u_fun(y)
            dU = du_fun(y)
            for k in range(n):

                def setMat(mat, eq, var, value):
                    jj, kk = local_to_global_index(j, k, eq, var)
                    mat[jj, kk] = value

                u = phi(k, 0, y)
                d2u = phi(k, 2, y)

                v = u
                dv = phi(k, 1, y)
                d2v = d2u

                w = u
                d2w = d2u

                p = cheb(k, 0, y)
                dp = cheb(k, 1, y)

                # eq 1 (momentum x)
                setMat(B, 0, 0, u)

                setMat(A, 0, 0, -I * alpha * U * u + 1 / Re * (d2u - kSq * u))
                setMat(A, 0, 1, v * -dU)
                setMat(A, 0, 3, -I * alpha * p)

                # eq 2 (momentum y)
                setMat(B, 1, 1, v)

                setMat(A, 1, 1, -I * alpha * U * v + 1 / Re * (d2v - kSq * v))
                setMat(A, 1, 3, -dp)

                # eq 3 (momentum z)
                setMat(B, 2, 2, w)

                setMat(A, 2, 2, -I * alpha * U * w + 1 / Re * (d2w - kSq * w))
                setMat(A, 2, 3, -I * beta * p)

                # eq 4 (continuity)
                setMat(A, 3, 0, I * alpha * u)
                setMat(A, 3, 1, dv)
                setMat(A, 3, 2, I * beta * w)

        self.A = A
        self.B = B
        return (A, B)

    def read_mat(self, file, key):
        return scipy.io.loadmat(file)[key]

    def calculate_eigenvalues(self):
        try:
            if None in [self.A, self.B]:
                self.assemble_matrix_fast()
        except ValueError:
            pass
        eigvals, eigvecs = eig(self.A, self.B)

        # scale any spurious eigenvalues out of the picture
        for j in range(len(eigvals)):
            if eigvals[j].real > 1:
                eigvals[j] = -1e12
        # sort eigenvals
        eevs = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]
        eevs = sorted(eevs, reverse=True, key=lambda x: x[0].real)
        self.eigenvalues = np.array([eev[0] for eev in eevs])
        self.eigenvectors = [eev[1] for eev in eevs]
        self.eigenvalues.dump(
            "fields/eigenvalues_Re_" + str(self.Re) + "_n_" + str(self.n)
        )
        return self.eigenvalues, self.eigenvectors

    def velocity_field(self, domain, mode=0):
        assert domain.number_of_dimensions == 3, "this only makes sense in 3D."
        self.n = len(domain.grid[1])
        if type(self.eigenvalues) == NoneType:
            print("calculating eigenvalues")
            self.calculate_eigenvalues()

        evec = self.eigenvectors[mode]

        u_vec, v_vec, w_vec, _ = np.split(evec, 4)

        def to_3d_field(eigenvector):
            phi_mat = np.zeros((self.n, self.n), dtype=np.complex128)
            for i in range(self.n):
                for k in range(self.n):
                    phi_mat[i, k] = phi(k, 0, self.ys[i])
            out = np.outer(
                np.exp(1j * self.alpha * domain.grid[0]), phi_mat @ eigenvector
            ).real
            out = np.tile(out, (len(domain.grid[2]), 1, 1))
            out = np.moveaxis(out, 0, -1)
            return out

        print("calculating velocity pertubations in 3D")
        u_field = Field(domain, to_3d_field(u_vec), name="velocity_pert_x")
        v_field = Field(domain, to_3d_field(v_vec), name="velocity_pert_y")
        w_field = Field(domain, to_3d_field(w_vec), name="velocity_pert_z")
        print("done calculating velocity pertubations in 3D")

        self.velocity_field_ = VectorField([u_field, v_field, w_field])
        return (u_field, v_field, w_field)

    def energy_over_time(self, domain, mode=0, eps=1.0):
        if type(self.velocity_field_) == NoneType:
            try:
                Nx = domain.number_of_cells(0)
                Ny = domain.number_of_cells(1)
                Nz = domain.number_of_cells(2)
                make_field_file_name = (
                    lambda field_name: field_name
                    + "_"
                    + str(self.Re)
                    + "_"
                    + str(Nx)
                    + "_"
                    + str(Ny)
                    + "_"
                    + str(Nz)
                )
                u = Field.FromFile(domain, make_field_file_name("u"), name="u_pert")
                v = Field.FromFile(domain, make_field_file_name("v"), name="v_pert")
                w = Field.FromFile(domain, make_field_file_name("w"), name="w_pert")
                self.velocity_field_ = VectorField([u, v, w])
            except FileNotFoundError:
                print("Fields not found, performing eigenvalue computation.")
                self.velocity_field(domain, mode)
        try:
            self.eigenvalues = np.load(
                "fields/eigenvalues_Re_" + str(self.Re) + "_n_" + str(self.n),
                allow_pickle=True,
            )
        except FileNotFoundError:
            self.calculate_eigenvalues()

        def out(t, dim=None):
            if type(dim) == NoneType:
                energy = 0
                for d in domain.all_dimensions():
                    energy += (
                        self.velocity_field_[d]
                        * (jnp.exp(self.eigenvalues[mode].real * t))
                    ).energy()
            else:
                energy = (
                    self.velocity_field_[dim]
                    * (jnp.exp(self.eigenvalues[mode].real * t))
                ).energy()
            return eps**2 * energy

        return (out, self.eigenvalues[mode])

    def print_welcome(self):
        print("starting linear stability calculation")  # TODO more info

    def post_process(self):
        pass

    def perform_calculation(self):
        self.print_welcome()
        print("Loading DNS data")
        t0 = timeit.default_timer()
        # self.load_dns_data()
        t1 = timeit.default_timer()
        print("(took " + str(t1 - t0) + " seconds)")
        print("Performing matrix assembly (matrices A and B)")
        self.assemble_matrix_fast()
        t2 = timeit.default_timer()
        print("(took " + str(t2 - t1) + " seconds)")
        print("Calculating eigenvalues and -vectors")
        self.calculate_eigenvalues()
        t3 = timeit.default_timer()
        print("(took " + str(t3 - t2) + " seconds)")
        t5 = timeit.default_timer()
        print("(took " + str(t5 - t4) + " seconds)")
        self.post_process()
        print("Done")
