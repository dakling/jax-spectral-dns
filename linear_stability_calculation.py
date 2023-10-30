#!/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
from numpy.linalg import svd
from scipy.linalg import eig, cholesky
from scipy.sparse.linalg import eigs
from scipy.integrate import quad, fixed_quad, simpson, quadrature
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

        self.S = None
        self.V = None

        self.make_field_file_name = (
            lambda domain_, field_name, mode: field_name
            + "_"
            + str(self.Re)
            + "_"
            + str(domain_.number_of_cells(0))
            + "_"
            + str(domain_.number_of_cells(1))
            + "_"
            + str(domain_.number_of_cells(2))
            + "_mode_"
            + str(mode)
        )

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
        np.array(self.eigenvectors).dump(
            "fields/eigenvectors_Re_" + str(self.Re) + "_n_" + str(self.n)
        )
        return self.eigenvalues, self.eigenvectors

    def velocity_field(self, domain, mode=0, recompute_partial=False, recompute_full=False):
        assert domain.number_of_dimensions == 3, "This only makes sense in 3D."
        recompute_partial = recompute_partial or recompute_full
        self.n = domain.number_of_cells(1)
        if recompute_full or type(self.eigenvalues) == NoneType:
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

        try:
            if recompute_partial==False:
                u_field = Field.FromFile(domain, self.make_field_file_name(domain, "u", mode), name="velocity_pert_x")
                v_field = Field.FromFile(domain, self.make_field_file_name(domain, "v", mode), name="velocity_pert_y")
                w_field = Field.FromFile(domain, self.make_field_file_name(domain, "w", mode), name="velocity_pert_z")
            else:
                raise FileNotFoundError() # a bit of a HACK?
        except FileNotFoundError:
            print("calculating velocity pertubations in 3D")
            u_field = Field(domain, to_3d_field(u_vec), name="velocity_pert_x")
            v_field = Field(domain, to_3d_field(v_vec), name="velocity_pert_y")
            w_field = Field(domain, to_3d_field(w_vec), name="velocity_pert_z")
            print("done calculating velocity pertubations in 3D")

        u_field.save_to_file(self.make_field_file_name(domain, "u", mode))
        v_field.save_to_file(self.make_field_file_name(domain, "v", mode))
        w_field.save_to_file(self.make_field_file_name(domain, "w", mode))

        self.velocity_field_ = VectorField([u_field, v_field, w_field])
        return (u_field, v_field, w_field)

    def energy_over_time(self, domain, mode=0, eps=1.0):
        if type(self.velocity_field_) == NoneType:
            try:
                Nx = domain.number_of_cells(0)
                Ny = domain.number_of_cells(1)
                Nz = domain.number_of_cells(2)
                u = Field.FromFile(domain, self.make_field_file_name(domain, "u", mode), name="velocity_pert_x")
                v = Field.FromFile(domain, self.make_field_file_name(domain, "v", mode), name="velocity_pert_y")
                w = Field.FromFile(domain, self.make_field_file_name(domain, "w", mode), name="velocity_pert_z")
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

    def calculate_transient_growth_svd(self, domain, T, number_of_modes, save=False, recompute=False):
        if type(self.eigenvalues) == NoneType or type(self.eigenvectors) == NoneType:
            try:
                if recompute:
                    raise FileNotFoundError()
                else:
                    self.eigenvalues = np.load(
                        "fields/eigenvalues_Re_" + str(self.Re) + "_n_" + str(self.n),
                        allow_pickle=True,
                    )
                    self.eigenvectors = np.load(
                        "fields/eigenvectors_Re_" + str(self.Re) + "_n_" + str(self.n),
                        allow_pickle=True,
                    )
            except FileNotFoundError:
                self.calculate_eigenvalues()
        evs = self.eigenvalues[:number_of_modes]
        evecs = self.eigenvectors[:number_of_modes]
        n = self.n

        def get_integral_coefficient(p, q):
            f = lambda y: phi(p, 0, y) * phi(q, 0, y)
            # out, _ = quad(f, -1, 1)
            # out, _ = fixed_quad(f, -1, 1, n=2)
            out, _ = quadrature(f, -1, 1, maxiter=100)
            # ys = domain.grid[1]
            # f_ = np.fromiter((f(y) for y in ys), ys.dtype)
            # out = simpson(f_)
            return out

        integ = np.zeros([n, n])
        for p in range(n):
            for q in range(p, n):
                integ[p, q] = get_integral_coefficient(p, q)
                integ[q, p] = integ[p, q]  # not needed right?
        C = np.zeros([number_of_modes, number_of_modes], dtype=complex)
        for j in range(number_of_modes):
            for k in range(j, number_of_modes):
                for p in range(n):
                    for q in range(n):
                        for block in range(3):
                            C[j, k] += (
                                np.conjugate(evecs[j][p + block * n])
                                * evecs[k][q + block * n]
                                * integ[p, q]
                            )
                C[k, j] = np.conjugate(C[j, k])
            C[j, j] = C[j, j].real  # just elminates O(10^-16) complex parts which bothers `chol'
        F = cholesky(C)
        Sigma = np.diag([np.exp(evs[i] * T) for i in range(number_of_modes)])
        USVh = svd(F @ Sigma @ np.linalg.inv(F), compute_uv=True)
        U = USVh[0]
        S = USVh[1]
        V = USVh[2].T
        print("check that this is zero: ", jnp.linalg.norm(F @ Sigma @ np.linalg.inv(F) - U @ jnp.diag(S) @ V.T))
        if save:
            self.S = S
            self.V = V
        return (S, V)

    def calculate_transient_growth_max_energy(self, domain, T, number_of_modes):
        S, _ = self.calculate_transient_growth_svd(
            domain, T, number_of_modes, save=False
        )
        return S[0] ** 2

    def calculate_transient_growth_initial_condition(self, domain, T, number_of_modes, recompute_partial=False, recompute_full=False):
        """Calcluate the initial condition that achieves maximum growth at time
        T. Uses cached values for velocity fields and eigenvalues/-vectors,
        however, recompute_partial=True forces recomputation of the velocity
        fields (but not of eigenvalues/-vectors) and  recompute_full=True forces
        recomputation of eigenvalues/eigenvectors as well as velocity fields."""
        recompute_partial = recompute_partial or recompute_full
        if recompute_full or type(self.V) == NoneType:
            _, self.V = self.calculate_transient_growth_svd(
                domain, T, number_of_modes, save=True, recompute=recompute_full
            )
        V = self.V

        u_0, v_0, w_0 = self.velocity_field(domain, 0, recompute_partial=recompute_partial, recompute_full=recompute_full)
        u = V[0, 0] * u_0
        v = V[0, 0] * v_0
        w = V[0, 0] * w_0

        ys1 = []
        for mode in range(0, number_of_modes):
            ys1.append(abs(V[mode,0]))

        print("max growth: ", self.S[0]**2)
        fig, ax = plt.subplots(1,1)
        xs = list(range(number_of_modes))
        ax.plot(xs, ys1, "o")
        ax.set_yscale("log", base=10)
        fig.savefig("plots/plot.pdf")
        # raise Exception("break")

        for mode in range(1, number_of_modes):
            print("mode ", mode, " of ", number_of_modes)
            kappa_i = V[mode, 0]

            u_inc, v_inc, w_inc = self.velocity_field(domain,
                                                      mode,
                                                      recompute_partial=recompute_partial,
                                                      recompute_full=recompute_full)
            u += kappa_i * u_inc
            v += kappa_i * v_inc
            w += kappa_i * w_inc
        return (u, v, w)

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
