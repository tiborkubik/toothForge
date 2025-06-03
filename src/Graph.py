import lapy
import math
import scipy
import trimesh
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import eigs, eigsh


class Graph:
    def __init__(self,
                 mesh_tri: trimesh.Trimesh,
                 k: int = 3,
                 L_approx_type: str = 'fem'
                 ) -> None:
        # Inputs
        self.mesh_tri = mesh_tri  # store mesh
        self.k = k  # number of spectral features to extract.
        self.L_approx_type = L_approx_type

        # Iterate over the points saving their 3d location.
        self.points = self.mesh_tri.vertices

        # Assign matrices that will be used for laplacian and eigen decomposition.
        self.L, self.M = self.get_laplacian()  # M is the max matrix
        self.eig_vals, self.eig_vecs, self.eig_vecs_inv, self.X = self.apply_decomposition()

    def get_laplacian(self):
        print(f'Computing Laplacian of shape {len(self.mesh_tri.vertices)} x {len(self.mesh_tri.vertices)}...')
        L, M = self.get_laplace_operator_approximation()

        return L, M

    def apply_decomposition(self):
        print(f'Computing top {self.k} eigenvectors and eigenvalues...')
        eig_vals, eig_vecs = eigs(self.L, k=self.k, sigma=1e-10, which='LR')

        eig_vals = np.real(eig_vals)
        eig_vecs = np.real(eig_vecs)

        sign_f = 1 - 2 * (eig_vecs[0, :] < 0)
        eig_vecs *= sign_f

        eig_vals = eig_vals.real[:]
        eig_vecs = eig_vecs[:, :]
        eig_vecs_inv = np.linalg.pinv(eig_vecs)

        X = eig_vecs @ np.diag(eig_vals ** (-0.5)).T

        return eig_vals, eig_vecs, eig_vecs_inv, X

    def get_laplace_operator_approximation(self) -> tuple[scipy.sparse.coo_matrix, scipy.sparse.diags]:
        """Computes a discrete approximation of the laplace-beltrami operator on
        a given mesh. The approximation is given by a Mass matrix A and a weight or stiffness matrix W

        Args:
            mesh (trimesh.Trimesh): Input mesh
            approx (str, optional): Laplace approximation to use See laplace.approx_methods()
            for possible values. Defaults to 'cotangens'.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple of sparse matrices (Stiffness, Mass)
        """
        if self.L_approx_type == 'fem':
            T = lapy.TriaMesh(self.mesh_tri.vertices, self.mesh_tri.faces)
            solver = lapy.Solver(T)
            return solver.stiffness, solver.mass
        else:
            W = self.build_laplace_approximation_matrix(self.mesh_tri, self.L_approx_type)
            M = self.build_mass_matrix(self.mesh_tri)
            return W, M

    def build_laplace_approximation_matrix(self,
                                           mesh: trimesh.Trimesh,
                                           approx: str = 'beltrami'
                                           ) -> scipy.sparse.coo_matrix:
        """Build the sparse mesh laplacian matrix of the given mesh M=(V, E).
        This is a positive semidefinite matrix C:

               w_ij                    if (i, j) in E
        C_ij = -sum_{j in N(i)} (w_ij) if i == j
                0                      otherwise
        here h is the average edge length

        Args:
            mesh (trimesh.Trimesh): Mesh used to compute the matrix C
            approx (str): Approximation type to use, must be in ['beltrami', 'cotangens', 'mesh']. Defaults to 'beltrami'.

        Returns:
            A sparse matrix of size (#vertices, #vertices) representing the discrete Laplace operator.
        """

        if approx == 'beltrami':
            return self.build_laplace_betrami_matrix(mesh)
        elif approx == 'cotangens':
            return self.build_cotangens_matrix(mesh)
        else:
            return self.build_mesh_laplace_matrix(mesh)

    @staticmethod
    def build_mass_matrix(mesh: trimesh.Trimesh) -> scipy.sparse.diags:
        """Build the sparse diagonal mass matrix for a given mesh

        Args:
            mesh (trimesh.Trimesh): Mesh to use.

        Returns:
            A sparse diagonal matrix of size (#vertices, #vertices).
        """
        areas = np.zeros(shape=(len(mesh.vertices)))
        for face, area in zip(mesh.faces, mesh.area_faces):
            areas[face] += area / 3.0

        return scipy.sparse.diags(areas)

    @staticmethod
    def build_laplace_betrami_matrix(mesh: trimesh.Trimesh) -> scipy.sparse.coo_matrix:
        """Build the sparse laplace beltrami matrix of the given mesh M=(V, E).
        This is a positive semidefinite matrix C:

               -1         if (i, j) in E
        C_ij =  deg(V(i)) if i == j
                0         otherwise

        Args:
            mesh (trimesh.Trimesh): Mesh used to compute the matrix C
        """
        n = len(mesh.vertices)
        IJ = np.concatenate([
            mesh.edges,
            [[i, i] for i in range(n)]
        ], axis=0)
        V = np.concatenate([
            [-1 for _ in range(len(mesh.edges))],
            mesh.vertex_degree
        ], axis=0)

        A = scipy.sparse.coo_matrix((V, (IJ[..., 0], IJ[..., 1])), shape=(n, n), dtype=np.float64)
        return A

    @staticmethod
    def build_cotangens_matrix(mesh: trimesh.Trimesh) -> scipy.sparse.coo_matrix:
        """Build the sparse cotangens weight matrix of the given mesh M=(V, E).
        This is a positive semidefinite matrix C:

               -0.5 * (tan(a) + tan(b))  if (i, j) in E
        C_ij = -sum_{j in N(i)} (C_ij)   if i == j
                0                        otherwise

        Args:
            mesh (trimesh.Trimesh): Mesh used to compute the matrix C

        Returns:
            A sparse matrix of size (#vertices, #vertices) representing the discrete Laplace operator.
        """
        n = len(mesh.vertices)
        ij = mesh.face_adjacency_edges
        ab = mesh.face_adjacency_unshared

        uv = mesh.vertices[ij]
        lr = mesh.vertices[ab]

        def cotan(v1, v2):
            return np.sum(v1 * v2) / np.linalg.norm(np.cross(v1, v2), axis=-1)

        ca = cotan(lr[:, 0] - uv[:, 0], lr[:, 0] - uv[:, 1])
        cb = cotan(lr[:, 1] - uv[:, 0], lr[:, 1] - uv[:, 1])

        wij = np.maximum(0.5 * (ca + cb), 0.0)

        I = []
        J = []
        V = []
        for idx, (i, j) in enumerate(ij):
            I += [i, j, i, j]
            J += [j, i, i, j]
            V += [-wij[idx], -wij[idx], wij[idx], wij[idx]]

        A = scipy.sparse.coo_matrix((V, (I, J)), shape=(n, n), dtype=np.float64)
        return A

    @staticmethod
    def build_mesh_laplace_matrix(mesh: trimesh.Trimesh) -> scipy.sparse.coo_matrix:
        """Build the sparse mesh laplacian matrix of the given mesh M=(V, E).
        This is a positive semidefinite matrix C:

               -1/(4pi*h^2) * e^(-||vi-vj||^2/(4h)) if (i, j) in E
        C_ij = -sum_{j in N(i)} (C_ij)              if i == j
                0                                   otherwise
        here h is the average edge length

        Args:
            mesh (trimesh.Trimesh): Mesh used to compute the matrix C

        Returns:
            A sparse matrix of size (#vertices, #vertices) representing the discrete Laplace operator.
        """
        n = len(mesh.vertices)
        h = np.mean(mesh.edges_unique_length)
        a = 1.0 / (4 * math.pi * h * h)
        wij = a * np.exp(-mesh.edges_unique_length ** 2 / (4.0 * h))
        I = []
        J = []
        V = []
        for idx, (i, j) in enumerate(mesh.edges_unique):
            I += [i, j, i, j]
            J += [j, i, i, j]
            V += [-wij[idx], -wij[idx], wij[idx], wij[idx]]

        A = scipy.sparse.coo_matrix((V, (I, J)), shape=(n, n), dtype=np.float64)
        return A
