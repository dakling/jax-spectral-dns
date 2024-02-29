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

# from importlib import reload
import sys

# try:
#     reload(sys.modules["cheb"])
# except:
#     if hasattr(sys, "ps1"):
#         pass
from cheb import cheb, phi, phi_s, phi_a, phi_pressure

# try:
#     reload(sys.modules["domain"])
# except:
#     if hasattr(sys, "ps1"):
#         pass
from domain import PhysicalDomain

# try:
#     reload(sys.modules["field"])
# except:
#     if hasattr(sys, "ps1"):
#         pass
from field import PhysicalField, VectorField

# try:
#     reload(sys.modules["equation"])
# except:
#     if hasattr(sys, "ps1"):
#         pass
from equation import Equation

NoneType = type(None)


class LinearStabilityCalculation:
    def __init__(self, Re=180.0, alpha=3.25, beta=0.0, n=50):
        self.Re = Re
        self.alpha = alpha
        self.beta = beta
        # self.n = int(n * PhysicalDomain.aliasing)  # chebychev resolution
        self.n = n  # chebychev resolution

        self.A = None
        self.B = None
        self.eigenvalues = None
        self.eigenvectors = None

        self.C = None
        self.growth = []

        # self.ys = [np.cos(np.pi * (2*(i+1)-1) / (2*self.n)) for i in range(self.n)] # gauss-lobatto points (SH2001, p. 488)
        domain = PhysicalDomain.create((n,), (False,))
        self.ys = domain.grid[0]

        self.velocity_field_ = None

        self.S = None
        self.V = None
        self.U = None

        self.symm = False
        # self.symm = True

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
        beta = self.beta
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
        # TODO cleaner to remove them? looking at further analysis
        clean_eigvals = []
        clean_eigvecs = []
        for j in range(len(eigvals)):
            # if eigvals[j].real > 1:
            #     eigvals[j] = -1e12
            #     eigvecs[j] = np.zeros_like(eigvecs[j])
            if eigvals[j].real <= 1:
                clean_eigvals.append(eigvals[j])
                clean_eigvecs.append(eigvecs[:, j])

        # sort eigenvals
        eevs = [(clean_eigvals[i], clean_eigvecs[i]) for i in range(len(clean_eigvals))]
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
        mode=0,
        factor=1.0,
        recompute_partial=False,
        recompute_full=False,
        save=True,
    ):
        assert domain.number_of_dimensions == 3, "This only makes sense in 3D."
        recompute_partial = recompute_partial or recompute_full
        try:
            if recompute_partial == False:
                u_field = PhysicalField.FromFile(
                    domain,
                    self.make_field_file_name_mode(domain, "u", mode),
                    name="velocity_pert_x",
                )
                v_field = PhysicalField.FromFile(
                    domain,
                    self.make_field_file_name_mode(domain, "v", mode),
                    name="velocity_pert_y",
                )
                w_field = PhysicalField.FromFile(
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

            u_vec, v_vec, w_vec, _ = jnp.split(evec, 4)

            N_domain = domain.number_of_cells(1)
            ys = domain.grid[1]

            def to_3d_field(eigenvector, component=0):
                phi_mat = jnp.zeros((N_domain, self.n), dtype=jnp.complex64)
                for i in range(N_domain):
                    for k in range(self.n):
                        if self.symm:
                            # phi_mat[i, k] = [phi_a, phi_s, phi_a][component](k, 0, ys[i])
                            phi_mat = phi_mat.at[i, k].set(
                                [phi_a, phi_s, phi_a][component](k, 0, ys[i])
                            )
                        else:
                            phi_mat = phi_mat.at[i, k].set(phi(k, 0, ys[i]))
                            # phi_mat[i, k] = phi(k, 0, ys[i])
                out = (
                    factor
                    * jnp.einsum(
                        "ij,k->ijk",
                        jnp.einsum(
                            "i,j->ij",
                            jnp.exp(
                                # 1j * self.alpha * domain.grid[0] + self.eigenvalues[mode] * time
                                1j
                                * (self.alpha * domain.grid[0])
                            ),
                            phi_mat @ eigenvector,
                        ),
                        jnp.exp(
                            1j * (self.beta * domain.grid[2]),
                        ),
                    )
                ).real

                # if abs(self.beta) < 1e-25:
                    # print("testing against old implementation")
                    # out_legacy = (factor * jnp.outer(
                    #             jnp.exp(
                    #                 # 1j * self.alpha * domain.grid[0] + self.eigenvalues[mode] * time
                    #                 1j
                    #                 * (self.alpha * domain.grid[0])
                    #             ),
                    #             phi_mat @ eigenvector,
                    #         )
                    # ).real
                    # out_legacy = jnp.tile(out_legacy, (len(domain.grid[2]), 1, 1))
                    # out_legacy = jnp.moveaxis(out_legacy, 0, -1)
                    # assert (out == out_legacy).all()
                return out

            print("calculating velocity perturbations in 3D")
            u_field = PhysicalField(
                domain, to_3d_field(u_vec, component=0), name="velocity_pert_x"
            )
            v_field = PhysicalField(
                domain, to_3d_field(v_vec, component=1), name="velocity_pert_y"
            )
            w_field = PhysicalField(
                domain, to_3d_field(w_vec, component=2), name="velocity_pert_z"
            )
            print("done calculating velocity perturbations in 3D")

            if save:
                u_field.save_to_file(self.make_field_file_name_mode(domain, "u", mode))
                v_field.save_to_file(self.make_field_file_name_mode(domain, "v", mode))
                w_field.save_to_file(self.make_field_file_name_mode(domain, "w", mode))

        self.velocity_field_ = VectorField([u_field, v_field, w_field])
        return self.velocity_field_

    def velocity_field_y_slice(self, domain, mode=0, factor=1.0, recompute_full=False):
        assert domain.number_of_dimensions == 3, "This only makes sense in 3D."
        if recompute_full or type(self.eigenvalues) == NoneType:
            print("calculating eigenvalues")
            self.calculate_eigenvalues()

        evec = self.eigenvectors[mode]

        u_vec, v_vec, w_vec, _ = np.split(evec, 4)

        N_domain = domain.number_of_cells(1)
        ys = domain.grid[1]

        def to_slice(eigenvector, component=0):
            if abs(self.beta) > 1e-25:
                raise Exception("Spanwise dependency not implemented yet.")
            phi_mat = np.zeros((N_domain, self.n), dtype=np.complex64)
            for i in range(N_domain):
                for k in range(self.n):
                    if self.symm:
                        phi_mat[i, k] = [phi_a, phi_s, phi_a][component](k, 0, ys[i])
                    else:
                        phi_mat[i, k] = phi(k, 0, ys[i])
            out = factor * phi_mat @ eigenvector
            return jnp.array(out.tolist())

        print("calculating velocity perturbations in 3D")
        u_slice = to_slice(u_vec, component=0)
        v_slice = to_slice(v_vec, component=1)
        w_slice = to_slice(w_vec, component=2)
        print("done calculating velocity perturbations in 3D")

        return (u_slice, v_slice, w_slice)

    def energy_over_time(self, domain, mode=0, eps=1.0):
        if type(self.velocity_field_) == NoneType:
            try:
                u = PhysicalField.FromFile(
                    domain,
                    self.make_field_file_name_mode(domain, "u", mode),
                    name="velocity_pert_x",
                )
                v = PhysicalField.FromFile(
                    domain,
                    self.make_field_file_name_mode(domain, "v", mode),
                    name="velocity_pert_y",
                )
                w = PhysicalField.FromFile(
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
                out_s, _ = quad(f_s, -1, 1, limit=100)
                out_a, _ = quad(f_a, -1, 1, limit=100)
                # xs = self.ys
                # fs_s = list(map(f_s, xs))
                # fa_s = list(map(f_a, xs))
                # out_s = simpson(fs_s, x=xs)
                # out_a = simpson(fa_s, x=xs)
                return (out_s, out_a)
            else:
                out, _ = quad(f, -1, 1, limit=100)
                # xs = self.ys
                # fs = np.fromiter(map(f, xs), dtype=np.float64)
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
                                    * [integ_a[p, q], integ_s[p, q], integ_a[p, q]][
                                        block
                                    ]
                                )
                            else:
                                C[j, k] += (
                                    np.conjugate(evecs[j][p + block * n])
                                    * evecs[k][q + block * n]
                                    * integ[p, q]
                                )
                C[k, j] = np.conjugate(C[j, k])
            C[j, j] = C[
                j, j
            ].real  # just elminates O(10^-16) complex parts which bothers `chol'
        F = cholesky(C)
        Sigma = np.diag([np.exp(evs[i] * T) for i in range(number_of_modes)])
        mat = F @ Sigma @ np.linalg.inv(F)
        U, S, Vh = svd(mat, compute_uv=True)
        V = Vh.conj().T
        if save:
            self.S = S
            self.U = U
            self.V = V

        return (S, V)

    def calculate_transient_growth_max_energy(self, domain, T, number_of_modes):
        S, _ = self.calculate_transient_growth_svd(
            domain, T, number_of_modes, save=False
        )
        return S[0] ** 2

    def calculate_transient_growth_initial_condition_from_coefficients(
        self, domain, coeffs, save=False, recompute=True
    ):
        """Calcluate the initial condition that achieves maximum growth at time
        T. Uses cached values for velocity fields and eigenvalues/-vectors,
        however, recompute_partial=True forces recomputation of the velocity
        fields (but not of eigenvalues/-vectors) and  recompute_full=True forces
        recomputation of eigenvalues/eigenvectors as well as velocity fields."""


        factors = coeffs
        u_0 = self.velocity_field(
            domain,
            0,
            save=save,
            recompute_full=recompute,
            recompute_partial=True,
            factor=factors[0],
        )
        u = u_0



        i = 1
        number_of_modes = len(coeffs)
        for mode in range(1, number_of_modes):
            factor = factors[mode]
            # if abs(factor) > 1e-8:
            if True:
                i += 1
                # print("mode", mode, "of", number_of_modes, "factor:", factor)
                print("mode", mode, "of", number_of_modes)
                u_inc = self.velocity_field(
                    domain,
                    mode,
                    save=save,
                    recompute_full=False,  # no need to recompute eigenvalues and eigenvectors
                    recompute_partial=True,
                    factor=factors[mode],
                )
                u += u_inc

            else:
                print("mode ", mode, " of ", number_of_modes, "(negligble, skipping)")

        print("modes used:", i)
        # print("energy of initial field:", u.energy())
        # print("expected energy growth: ", self.S[0]**2)


        return u

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
                u_ = PhysicalField.FromFile(
                    domain,
                    self.make_field_file_name(domain, "u"),
                    name="velocity_pert_x",
                )
                v_ = PhysicalField.FromFile(
                    domain,
                    self.make_field_file_name(domain, "v"),
                    name="velocity_pert_y",
                )
                w_ = PhysicalField.FromFile(
                    domain,
                    self.make_field_file_name(domain, "w"),
                    name="velocity_pert_z",
                )
                u = VectorField([u_, v_, w_])
            else:
                raise FileNotFoundError()  # a bit of a HACK?
        except FileNotFoundError:
            if recompute_full or type(self.V) == NoneType:
                self.S, self.V = self.calculate_transient_growth_svd(
                    domain, T, number_of_modes, save=True, recompute=recompute_full
                )
            U = self.U
            V = self.V

            print("expected energy growth: ", self.S[0] ** 2)

            factors = V[:, 0]
            # TODO use calculate_transient_growth_initial_condition_from_coefficients

            # final_factors = U[:,0] * self.S[0]
            # u_0_slice, v_0_slice, w_0_slice = self.velocity_field_y_slice(
            #     domain,
            #     0,
            #     recompute_full=recompute_full,
            #     # factor=U[0,0]
            #     factor=V[0,0]
            # )
            u_0 = self.velocity_field(
                domain,
                0,
                recompute_partial=recompute_partial,
                recompute_full=recompute_full,
                save=save_modes,
                factor=factors[0],
            )
            u = u_0
            # u_final_0 = self.velocity_field(
            #     domain,
            #     0,
            #     recompute_partial=recompute_partial,
            #     recompute_full=recompute_full,
            #     save=save_modes,
            #     factor=final_factors[0]
            # )
            # u_final = u_final_0
            # print("factor(0):", factors[0])
            # print("energy(0):", abs(u_0.energy()/factors[0]**2))
            # u_0.set_name("u_" + str(0))
            # u_0.plot_3d(2)
            # u_slice = u_0_slice
            # v_slice = v_0_slice
            # w_slice = w_0_slice

            fig_eig, ax_eig = plt.subplots(1, 1)
            ax_eig.plot(-self.eigenvalues.imag, self.eigenvalues.real, "k.", alpha=0.4)
            ax_eig.plot(
                -self.eigenvalues.imag[:number_of_modes],
                self.eigenvalues[:number_of_modes].real,
                "r.",
                alpha=0.4,
            )
            ax_eig.set_xlim([0.2, 1.0])
            ax_eig.set_ylim([-1.0, 0.0])
            ax_eig.plot([-self.eigenvalues[0].imag], [self.eigenvalues[0].real], "ko")

            n = [0]
            energy = abs(u_0.energy() / factors[0] ** 2)
            ys = [abs(factors[0] * energy)]
            i = 1
            index = 1
            for mode in range(1, number_of_modes):
                factor = factors[mode]
                if abs(factor) > 1e-8:
                    i += 1
                    print("mode", mode, "of", number_of_modes, "factor:", factor)
                    # u_slice_inc, v_slice_inc, w_slice_inc = self.velocity_field_y_slice(
                    #     domain,
                    #     mode,
                    #     factor=factor,
                    #     recompute_full=recompute_full
                    # )
                    u_inc = self.velocity_field(
                        domain,
                        mode,
                        recompute_partial=recompute_partial,
                        recompute_full=recompute_full,
                        save=save_modes,
                        factor=factors[mode],
                    )
                    # u_inc.set_name("u_" + str(mode))
                    # u_inc.plot_3d(2)
                    u += u_inc
                    # u_slice += u_slice_inc
                    # v_slice += v_slice_inc
                    # w_slice += w_slice_inc
                    energy = abs(u_inc.energy() / factors[mode] ** 2)
                    print("energy:", energy)

                    # u_final_inc = self.velocity_field(
                    #     domain,
                    #     mode,
                    #     recompute_partial=recompute_partial,
                    #     recompute_full=recompute_full,
                    #     save=save_modes,
                    #     factor=final_factors[mode] * self.S[0]
                    # )
                    # u_final += u_final_inc
                    # u_ = u
                    # u_.set_name("u_after_" + str(mode))
                    # u_.plot_3d(2)
                    # if abs(factor) > 1e-7:
                    #     if u_inc.energy() / u_0.energy() > 1e2:
                    #         print("high energy mode encountered: ", u_inc.energy() / u_0.energy())

                    # n.append(mode)
                    n.append(index)
                    index += 1
                    ys.append(abs(factors[mode] * energy))

                    ax_eig.plot(
                        [-self.eigenvalues[mode].imag],
                        [self.eigenvalues[mode].real],
                        "ko",
                    )
                else:
                    print(
                        "mode ", mode, " of ", number_of_modes, "(negligble, skipping)"
                    )
                    ax_eig.plot(
                        [-self.eigenvalues[mode].imag],
                        [self.eigenvalues[mode].real],
                        "rx",
                    )

                fig_eig.savefig("./plots/eigenvalues.pdf")

                rh_93_coeffs = np.genfromtxt("rh93_coeffs.csv", delimiter=",").T
                try:
                    # v_mat_ns = self.read_mat("v.mat", "v_s")[:number_of_modes, 0]
                    # u_mat_ns = self.read_mat("u.mat", "u_s")[:number_of_modes, 0]
                    # v_mat = self.read_mat("v_ns.mat", "v_s")[:number_of_modes, 0]
                    # u_mat = self.read_mat("u_ns.mat", "u_s")[:number_of_modes, 0]
                    # v_mat_full = self.read_mat("v_full.mat", "v_s")[:number_of_modes, 0]
                    # u_mat_full = self.read_mat("u_full.mat", "u_s")[:number_of_modes, 0]
                    fig, ax = plt.subplots(1, 2)
                    ax[0].set_yscale("log")
                    ax[0].plot(n, np.array(ys) / ys[0], "o")
                    ax[0].plot(
                        rh_93_coeffs[0], rh_93_coeffs[1] / rh_93_coeffs[1][0], "x"
                    )
                    # ax[0][0].plot(n, v_mat, "x")
                    # ax[0][0].plot(n, v_mat_full, ".")
                    # ax[0][0].plot(n, v_mat_ns, "x")
                    ax[1].set_ylim([1e-4, 1e4])
                    ax[1].set_yscale("log")
                    ax[1].plot(n, np.array(ys) / ys[0], "o")
                    ax[1].plot(
                        rh_93_coeffs[0], rh_93_coeffs[1] / rh_93_coeffs[1][0], "x"
                    )
                    fig.savefig("plots/coeffs.pdf")
                except Exception:
                    raise Exception()

            print("modes used:", i)
            print("energy of initial field:", u.energy())
            # print("energy of final field:", u_final.energy())
            # print("gain:", u_final.energy() / u.energy())

            # def slice_to_3d_field(slice):
            #     if abs(self.beta) > 1e-25:
            #         raise Exception("Spanwise dependency not implemented yet.")
            #     out = (np.outer(
            #         np.exp(
            #             # 1j * self.alpha * domain.grid[0] + self.eigenvalues[mode] * time
            #             1j * (self.alpha * domain.grid[0])
            #         ),
            #         slice,
            #     )).real
            #     out = np.tile(out, (len(domain.grid[2]), 1, 1))
            #     out = np.moveaxis(out, 0, -1)
            #     return jnp.array(out.tolist())

            # u = VectorField([Field(domain, slice_to_3d_field(slice), name="velocity_" + "xyz"[i]) for i, slice in enumerate([u_slice, v_slice, w_slice])])

            if save_final:
                for i in range(len(u)):
                    u[i].name = "velocity_perturbation_" + "xyz"[i]
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
