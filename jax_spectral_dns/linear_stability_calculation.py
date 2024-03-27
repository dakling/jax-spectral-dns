#!/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
from numpy.linalg import  svd
from scipy.linalg import eig, cholesky
from scipy.integrate import quad
import scipy
import timeit

from jax_spectral_dns.cheb import cheb, phi, phi_s, phi_a, phi_pressure
from jax_spectral_dns.domain import PhysicalDomain
from jax_spectral_dns.equation import print_verb
from jax_spectral_dns.field import PhysicalField, VectorField

NoneType = type(None)


class LinearStabilityCalculation:
    def __init__(self, Re=180.0, alpha=3.25, beta=0.0, n=50):
        self.Re = Re
        self.alpha = alpha
        self.beta = beta
        self.n = n  # chebychev resolution

        self.A = None
        self.B = None
        self.eigenvalues = None
        self.eigenvectors = None

        self.C = None
        self.growth = []

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

        clean_eigvals = []
        clean_eigvecs = []
        for j in range(len(eigvals)):
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


    def velocity_field_single_mode(
            self,
            domain,
            mode=0,
            factor=1.0,
            recompute_partial=False,
            recompute_full=False,
            save=True,
    ):
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
                u = VectorField([u_field, v_field, w_field])
            else:
                raise FileNotFoundError()  # a bit of a HACK?
        except FileNotFoundError:
            if recompute_full or type(self.eigenvalues) == NoneType:
                print_verb("calculating eigenvalues", verbosity_level=2)
                self.calculate_eigenvalues()
            u = self.velocity_field(domain, self.eigenvectors[mode], factor)

            if save:
                u[0].save_to_file(self.make_field_file_name_mode(domain, "u", mode))
                u[1].save_to_file(self.make_field_file_name_mode(domain, "v", mode))
                u[2].save_to_file(self.make_field_file_name_mode(domain, "w", mode))
        return u

    def velocity_field(
            self,
            domain,
            evec,
            factor=1.0,
            symm=False
    ):
        assert domain.number_of_dimensions == 3, "This only makes sense in 3D."

        u_vec, v_vec, w_vec, _ = jnp.split(evec, 4)

        N_domain = domain.number_of_cells(1)
        ys = domain.grid[1]

        n = u_vec.shape[0]
        # phi_mat = jnp.zeros((N_domain, self.n), dtype=jnp.float64)
        phi_mat = jnp.zeros((N_domain, n), dtype=jnp.float64)
        if not symm:
            for i in range(N_domain):
                for k in range(n):
                    phi_mat = phi_mat.at[i, k].set(phi(k, 0, ys[i]))
        else:
            for i in range(N_domain):
                for k in range(n):
                    phi_mat = phi_mat.at[i, k].set(phi_s(k, 0, ys[i]))

        return self.velocity_field_from_y_slice(domain, (phi_mat @ u_vec, phi_mat @ v_vec, phi_mat @ w_vec), factor=factor)

    def y_slice_to_3d_field(self, domain, y_slice_i, factor=1.0):
        out = (
            factor
            * jnp.einsum(
                "ij,k->ijk",
                jnp.einsum(
                    "i,j->ij",
                    jnp.exp(
                        1j
                        * (self.alpha * domain.grid[0])
                    ),
                    y_slice_i,
                ),
                jnp.exp(
                    1j * (self.beta * domain.grid[2]),
                ),
            )
        ).real

        return out

    def velocity_field_from_y_slice(
            self,
            domain,
            y_slice,
            factor=1.0
    ):
        assert domain.number_of_dimensions == 3, "This only makes sense in 3D."

        u_vec = y_slice[0]
        v_vec = y_slice[1]
        w_vec = y_slice[2]


        u_field = PhysicalField(
            domain, self.y_slice_to_3d_field(domain, u_vec, factor), name="velocity_pert_x"
        )
        v_field = PhysicalField(
            domain, self.y_slice_to_3d_field(domain, v_vec, factor), name="velocity_pert_y"
        )
        w_field = PhysicalField(
            domain, self.y_slice_to_3d_field(domain, w_vec, factor), name="velocity_pert_z"
        )

        self.velocity_field_ = VectorField([u_field, v_field, w_field])
        return self.velocity_field_

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
                print_verb("Fields not found, performing eigenvalue computation.", verbosity_level=2)
                self.velocity_field(domain, self.eigenvector[mode], save=False)
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
                out_s, _ = quad(f_s, -1, 1, limit=100)
                out_a, _ = quad(f_a, -1, 1, limit=100)
                return (out_s, out_a)
            else:
                out, _ = quad(f, -1, 1, limit=100)
                return out

        integ = np.zeros([n, n])
        integ_s = np.zeros([n, n])
        integ_a = np.zeros([n, n])
        for p in range(n):
            for q in range(p, n):
                if self.symm:
                    integ_s[p, q], integ_a[p, q] = get_integral_coefficient(p, q)
                    integ_s[q, p] = integ_s[p, q]
                    integ_a[q, p] = integ_a[p, q]
                else:
                    integ[p, q] = get_integral_coefficient(p, q)
                    integ[q, p] = integ[p, q]
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
            # elminate O(10^-16) complex parts which bothers `chol'
            C[j, j] = C[j, j].real
        F = cholesky(C, lower=False)
        F_inv = np.linalg.inv(F)
        Sigma = np.diag([np.exp(evs[i] * T) for i in range(number_of_modes)])
        mat = F @ Sigma @ F_inv
        U, S, Vh = svd(mat, compute_uv=True)
        V = Vh.conj().T
        coeffs = F_inv @ V[:, 0]
        if save:
            self.S = S
            self.U = U
            self.V = V

        return (S, coeffs)

    def calculate_transient_growth_max_energy(self, domain, T, number_of_modes):
        S, _ = self.calculate_transient_growth_svd(
            domain, T, number_of_modes, save=False
        )
        return S[0] ** 2

    def calculate_transient_growth_initial_condition_from_coefficients(
        self, domain, coeffs, recompute=True
    ):
        """Calcluate the initial condition that achieves maximum growth at time
        T. Uses cached values for eigenvalues/-vectors,
        however, recompute=True forces recomputation."""

        if recompute or type(self.eigenvectors) == NoneType:
            self.calculate_eigenvalues()

        factors = coeffs

        number_of_modes = len(factors)
        combined_ev = (np.array(self.eigenvectors[:number_of_modes])).T @ factors
        u = self.velocity_field(domain, combined_ev)

        return u

    def calculate_transient_growth_initial_condition(
        self,
        domain,
        T,
        number_of_modes,
        recompute_partial=True,
        recompute_full=True,
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
                self.S, factors = self.calculate_transient_growth_svd(
                    domain, T, number_of_modes, save=True, recompute=recompute_full
                )


            u = self.calculate_transient_growth_initial_condition_from_coefficients(domain, factors, recompute_full)

            if save_final:
                for i in range(len(u)):
                    u[i].name = "velocity_perturbation_" + "xyz"[i]
                    u[i].save_to_file(self.make_field_file_name(domain, "uvw"[i]))

        return u

    def print_welcome(self):
        print_verb("starting linear stability calculation", verbosity_level=2)

    def post_process(self):
        pass

    def perform_calculation(self):
        self.print_welcome()
        print_verb("Loading DNS data", verbosity_level=2)
        t0 = timeit.default_timer()
        # self.load_dns_data()
        t1 = timeit.default_timer()
        print_verb("(took " + str(t1 - t0) + " seconds)", verbosity_level=2)
        print_verb("Performing matrix assembly (matrices A and B)", verbosity_level=2)
        self.assemble_matrix_fast()
        t2 = timeit.default_timer()
        print_verb("(took " + str(t2 - t1) + " seconds)", verbosity_level=2)
        print_verb("Calculating eigenvalues and -vectors", verbosity_level=2)
        self.calculate_eigenvalues()
        t3 = timeit.default_timer()
        print_verb("(took " + str(t3 - t2) + " seconds)", verbosity_level=2)
        t4 = timeit.default_timer()
        print_verb("(took " + str(t4 - t3) + " seconds)", verbosity_level=2)
        self.post_process()
        print_verb("Done", verbosity_level=2)
