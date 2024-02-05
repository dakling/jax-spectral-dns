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

try:
    reload(sys.modules["cheb"])
except:
    if hasattr(sys, "ps1"):
        pass
from cheb import cheb, phi, phi_s, phi_a, phi_pressure

try:
    reload(sys.modules["domain"])
except:
    if hasattr(sys, "ps1"):
        pass
from domain import Domain

try:
    reload(sys.modules["field"])
except:
    if hasattr(sys, "ps1"):
        pass
from field import Field, VectorField

try:
    reload(sys.modules["equation"])
except:
    if hasattr(sys, "ps1"):
        pass
from equation import Equation

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
        self.U = None

        self.symm = False # TODO possibly make this nicer

        self.make_field_file_name_mode = (
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
        self.make_field_file_name = (
            lambda domain_, field_name: field_name
            + "_"
            + str(self.Re)
            + "_"
            + str(domain_.number_of_cells(0))
            + "_"
            + str(domain_.number_of_cells(1))
            + "_"
            + str(domain_.number_of_cells(2))
        )
        # Equation.initialize()

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

                if self.symm:
                    u = phi_a(k, 0, y)
                    d2u = phi_a(k, 2, y)

                    v = u
                    dv = phi_s(k, 1, y)
                    d2v = d2u

                    w = u
                    d2w = d2u

                    p = phi_pressure(k, 0, y)
                    dp = phi_pressure(k, 1, y)
                else:
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

    def velocity_field(
        self,
        domain,
        time=0.0,
        mode=0,
        recompute_partial=False,
        recompute_full=False,
        save=True,
    ):
        assert domain.number_of_dimensions == 3, "This only makes sense in 3D."
        recompute_partial = recompute_partial or recompute_full
        try:
            if recompute_partial == False and time <= 1e-15:
                u_field = Field.FromFile(
                    domain,
                    self.make_field_file_name_mode(domain, "u", mode),
                    name="velocity_pert_x",
                )
                v_field = Field.FromFile(
                    domain,
                    self.make_field_file_name_mode(domain, "v", mode),
                    name="velocity_pert_y",
                )
                w_field = Field.FromFile(
                    domain,
                    self.make_field_file_name_mode(domain, "w", mode),
                    name="velocity_pert_z",
                )
            else:
                raise FileNotFoundError()  # a bit of a HACK?
        except FileNotFoundError:
            if recompute_full or type(self.eigenvalues) == NoneType:
                print("calculating eigenvalues")
                self.calculate_eigenvalues()

            evec = self.eigenvectors[mode]

            u_vec, v_vec, w_vec, _ = np.split(evec, 4)

            N_domain = domain.number_of_cells(1)
            ys = domain.grid[1]
            def to_3d_field(eigenvector):
                phi_mat = np.zeros((N_domain, self.n), dtype=np.complex128)
                for i in range(N_domain):
                    for k in range(self.n):
                        phi_mat[i, k] = phi(k, 0, ys[i]) # TODO
                    #     phi_mat[i, k] = [phi_a, phi_s, phi_a][k](k, 0, ys[i]) # TODO
                out = np.outer(
                    np.exp(
                        1j * self.alpha * domain.grid[0] + self.eigenvalues[mode] * time
                    ),
                    phi_mat @ eigenvector,
                ).real
                out = np.tile(out, (len(domain.grid[2]), 1, 1))
                out = np.moveaxis(out, 0, -1)
                return jnp.array(out.tolist())

            print("calculating velocity pertubations in 3D")
            u_field = Field(domain, to_3d_field(u_vec), name="velocity_pert_x")
            v_field = Field(domain, to_3d_field(v_vec), name="velocity_pert_y")
            w_field = Field(domain, to_3d_field(w_vec), name="velocity_pert_z")
            print("done calculating velocity pertubations in 3D")

            if save:
                u_field.save_to_file(self.make_field_file_name_mode(domain, "u", mode))
                v_field.save_to_file(self.make_field_file_name_mode(domain, "v", mode))
                w_field.save_to_file(self.make_field_file_name_mode(domain, "w", mode))

        self.velocity_field_ = VectorField([u_field, v_field, w_field])
        return self.velocity_field_

    def energy_over_time(self, domain, mode=0, eps=1.0):
        if type(self.velocity_field_) == NoneType:
            try:
                u = Field.FromFile(
                    domain,
                    self.make_field_file_name_mode(domain, "u", mode),
                    name="velocity_pert_x",
                )
                v = Field.FromFile(
                    domain,
                    self.make_field_file_name_mode(domain, "v", mode),
                    name="velocity_pert_y",
                )
                w = Field.FromFile(
                    domain,
                    self.make_field_file_name_mode(domain, "w", mode),
                    name="velocity_pert_z",
                )
                self.velocity_field_ = VectorField([u, v, w])
            except FileNotFoundError:
                print("Fields not found, performing eigenvalue computation.")
                self.velocity_field(domain, mode, save=False)
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

    def calculate_transient_growth_svd(
        self, domain, T, number_of_modes, save=False, recompute=False
    ):
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
            f_s = lambda y: phi_s(p, 0, y) * phi_s(q, 0, y)
            f_a = lambda y: phi_a(p, 0, y) * phi_a(q, 0, y)
            if self.symm:
                # out_s, _ = fixed_quad(f_s, -1, 1, n=2)
                # out_a, _ = fixed_quad(f_a, -1, 1, n=2)
                # out_s, _ = quad(f_s, -1, 1, limit=100)
                # out_a, _ = quad(f_a, -1, 1, limit=100)
                xs = self.ys
                fs_s = list(map(f_s, xs))
                fa_s = list(map(f_a, xs))
                out_s = simpson(fs_s, x=xs)
                out_a = simpson(fa_s, x=xs)
                return (out_s, out_a)
            else:
                out, _ = quad(f, -1, 1, limit=100)
                # xs = self.ys
                # fs = list(map(f, xs))
                # out = simpson(fs, x=xs)
                return out

        integ = np.zeros([n, n])
        integ_s = np.zeros([n, n])
        integ_a = np.zeros([n, n])
        for p in range(n):
            for q in range(p, n):
                if self.symm:
                    integ_s[p, q], integ_a[p, q] = get_integral_coefficient(p, q)
                    integ_s[q, p] = integ_s[p, q]  # not needed right?
                    integ_a[q, p] = integ_a[p, q]  # not needed right?
                else:
                    integ[p, q] = get_integral_coefficient(p, q)
                    integ[q, p] = integ[p, q]  # not needed right?
        C = np.zeros([number_of_modes, number_of_modes], dtype=complex)
        for j in range(number_of_modes):
            for k in range(j, number_of_modes):
                for p in range(n):
                    for q in range(n):
                        for block in range(3):
                            if self.symm:
                                C[j, k] += (
                                    np.conjugate(evecs[j][p + block * n])
                                    * evecs[k][q + block * n]
                                    * [integ_a[p, q], integ_s[p, q], integ_a[p, q]][block]
                                )
                            else:
                                C[j, k] += (
                                    np.conjugate(evecs[j][p + block * n])
                                    * evecs[k][q + block * n]
                                    * integ[p, q]
                                )
                C[k, j] = np.conjugate(C[j, k])
            C[j, j] = C[j, j].real  # just elminates O(10^-16) complex parts which bothers `chol'
        F = cholesky(C)
        Sigma = np.diag([np.exp(evs[i] * T) for i in range(number_of_modes)])
        U, S, Vh = svd(F @ Sigma @ np.linalg.inv(F), compute_uv=True)
        V = Vh.T
        if save:
            self.S = S
            self.U = U
            self.V = V
        return (S, U)

    def calculate_transient_growth_max_energy(self, domain, T, number_of_modes):
        S, _ = self.calculate_transient_growth_svd(
            domain, T, number_of_modes, save=False
        )
        return S[0] ** 2

    def calculate_transient_growth_initial_condition(
        self,
        domain,
        T,
        number_of_modes,
        recompute_partial=False,
        recompute_full=False,
        save_modes=True,
        save_final=False,
    ):
        """Calcluate the initial condition that achieves maximum growth at time
        T. Uses cached values for velocity fields and eigenvalues/-vectors,
        however, recompute_partial=True forces recomputation of the velocity
        fields (but not of eigenvalues/-vectors) and  recompute_full=True forces
        recomputation of eigenvalues/eigenvectors as well as velocity fields."""

        recompute_partial = recompute_partial or recompute_full

        try:
            if recompute_partial is False:
                u_ = Field.FromFile(
                    domain,
                    self.make_field_file_name(domain, "u"),
                    name="velocity_pert_x",
                )
                v_ = Field.FromFile(
                    domain,
                    self.make_field_file_name(domain, "v"),
                    name="velocity_pert_y",
                )
                w_ = Field.FromFile(
                    domain,
                    self.make_field_file_name(domain, "w"),
                    name="velocity_pert_z",
                )
                u = VectorField([u_, v_, w_])
            else:
                raise FileNotFoundError()  # a bit of a HACK?
        except FileNotFoundError:
            if recompute_full or type(self.V) == NoneType:
                _, self.U = self.calculate_transient_growth_svd(
                    domain, T, number_of_modes, save=True, recompute=recompute_full
                )
            U = self.U
            V = self.V

            u_0 = self.velocity_field(
                domain,
                0,
                recompute_partial=recompute_partial,
                recompute_full=recompute_full,
                save=save_modes,
            )
            # u = u_0 * V[0, 0]
            u = u_0 * U[0, 0]

            n = []
            ys1 = []
            ys2 = []
            for mode in range(0, number_of_modes):
                n.append(mode)
                ys1.append(abs(V[mode, 0]))
                # ys1.append((V[mode, 0]))
                ys2.append(abs(U[mode, 0]))

            if True: #self.symm:
                v_mat_ns = self.read_mat("v.mat", "v_s")[:, 0]
                u_mat_ns = self.read_mat("u.mat", "u_s")[:, 0]
                v_mat = self.read_mat("v_ns.mat", "v_s")[:, 0]
                u_mat = self.read_mat("u_ns.mat", "u_s")[:, 0]
                v_mat_full = self.read_mat("v_full.mat", "v_s")[:, 0]
                u_mat_full = self.read_mat("u_full.mat", "u_s")[:, 0]
            else:
                pass
                # v_mat = self.read_mat("/home/klingenberg/Downloads/v_ns.mat", "v_s")[:, 0]
                # u_mat = self.read_mat("/home/klingenberg/Downloads/u_ns.mat", "u_s")[:, 0]
            print("shape u_mat: ", u_mat.shape)
            print("shape v_mat: ", v_mat.shape)
            fig, ax = plt.subplots(2,2)
            ax[0][0].set_yscale('log')
            ax[0][0].plot(n, ys1, "o")
            ax[0][0].plot(n, v_mat, "x")
            ax[0][0].plot(n, v_mat_full, ".")
            ax[0][0].plot(n, v_mat_ns, ".")
            ax[0][1].set_yscale('log')
            ax[0][1].plot(n, ys2, "o")
            ax[0][1].plot(n, u_mat, "x")
            ax[0][1].plot(n, u_mat_full, ".")
            ax[0][1].plot(n, u_mat_ns, ".")
            ax[1][0].set_ylim([1e-8, 1])
            ax[1][0].set_yscale('log')
            ax[1][0].plot(n, ys1, "o")
            ax[1][0].plot(n, v_mat, "x")
            ax[1][0].plot(n, v_mat_full, ".")
            ax[1][0].plot(n, v_mat_ns, ".")
            ax[1][1].set_yscale('log')
            ax[1][1].set_ylim([1e-13, 1])
            ax[1][1].plot(n, ys2, "o")
            ax[1][1].plot(n, u_mat, "x")
            ax[1][1].plot(n, u_mat_full, ".")
            ax[1][1].plot(n, u_mat_ns, ".")
            fig.savefig("plots/coeffs.pdf")
            print("energy growth: ", self.S[0]**2)
            if self.symm:
                raise Exception("break")

            for mode in range(1, number_of_modes):
                print("mode ", mode, " of ", number_of_modes)

                u_inc = self.velocity_field(
                    domain,
                    mode,
                    recompute_partial=recompute_partial,
                    recompute_full=recompute_full,
                    save=save_modes,
                )
                # u += u_inc * V[mode, 0]
                u += u_inc * U[mode, 0]

            if save_final:
                for i in range(len(u)):
                    u[i].name = "velocity_pertubation_" + "xyz"[i]
                    u[i].save_to_file(self.make_field_file_name(domain, "uvw"[i]))

        return u

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
        t4 = timeit.default_timer()
        print("(took " + str(t4 - t3) + " seconds)")
        self.post_process()
        print("Done")
