import os
import tables
import trimesh
import numpy as np
import matplotlib.pyplot as plt

from Graph import Graph
import pyvista as pv


class SpectralMesh:
    def __init__(self,
                 for_mesh_build: str | trimesh.Trimesh,
                 k: int
                 ) -> None:
        if isinstance(for_mesh_build, str):
            self.mesh_tri = trimesh.load(for_mesh_build, force='mesh')
        elif isinstance(for_mesh_build, trimesh.Trimesh):
            self.mesh_tri = for_mesh_build
        else:
            raise NotImplementedError

        print(f'Loaded mesh with {len(self.mesh_tri.vertices)} vertices and {len(self.mesh_tri.faces)} faces.')

        # Moving to origin and rescaling the mesh. Not necessary, just a normalization for a convenient processing.
        tform = [
            -(self.mesh_tri.bounds[1][i] + self.mesh_tri.bounds[0][i]) / 2.
            for i in range(3)
        ]
        matrix = np.eye(4)
        matrix[:3, 3] = tform
        self.mesh_tri.apply_transform(matrix)
        rescale = max(self.mesh_tri.extents) / 2.
        matrix = np.eye(4)
        matrix[:3, :3] /= rescale
        self.mesh_tri.apply_transform(matrix)

        self.coords, self.faces = self.mesh_tri.vertices, self.mesh_tri.faces

        self.k = k
        self.graph = Graph(self.mesh_tri, self.k)

        self.extended_data = None
        # self.extended_data = wave_signatures(self.graph.eig_vals, self.graph.eig_vecs, dim=32)

    def store_data(self,
                   target_f_p: str,
                   R: np.ndarray | None = None,
                   is_template: bool = False,
                   ) -> None:
        os.makedirs(target_f_p, exist_ok=True)
        self.mesh_tri.export(os.path.join(target_f_p, 'mesh.stl'))

        with tables.open_file(os.path.join(target_f_p, 'spectrum.h5'), mode='w') as handle:
            content_evals = handle.create_carray(
                handle.root,
                'evals',
                tables.Atom.from_dtype(self.graph.eig_vals.dtype),
                shape=self.graph.eig_vals.shape)
            content_evals[:] = self.graph.eig_vals

            content_evecs = handle.create_carray(
                handle.root,
                'evecs',
                tables.Atom.from_dtype(self.graph.eig_vecs.dtype),
                shape=self.graph.eig_vecs.shape)
            content_evecs[:] = self.graph.eig_vecs

        with tables.open_file(os.path.join(target_f_p, 'spec_coeff.h5'), mode='w') as handle:
            c = np.asarray(self.graph.eig_vecs.T @ self.mesh_tri.vertices)
            content_c = handle.create_carray(
                handle.root,
                'c',
                tables.Atom.from_dtype(c.dtype),
                shape=c.shape)
            content_c[:] = c

        with tables.open_file(os.path.join(target_f_p, 'spatial.h5'), mode='w') as handle:
            content_coords = handle.create_carray(
                handle.root,
                'coords',
                tables.Atom.from_dtype(self.coords.dtype),
                shape=self.coords.shape)
            content_coords[:] = self.coords

            content_faces = handle.create_carray(
                handle.root,
                'faces',
                tables.Atom.from_dtype(self.faces.dtype),
                shape=self.faces.shape)
            content_faces[:] = self.faces

        if not is_template:
            # Store mesh with geometry reconstructed from aligned and unaligned meshes.
            with tables.open_file(os.path.join(target_f_p, 'spec_rotation.h5'), mode='w') as handle:
                content_coords = handle.create_carray(
                    handle.root,
                    'R',
                    tables.Atom.from_dtype(R.dtype),
                    shape=R.shape)
                content_coords[:] = R

    def map_eigenmode_to_mesh_color(self,
                                    eigenmode_idx: int = 1
                                    ) -> None:
        target_eigenmode = self.graph.eig_vecs[:, eigenmode_idx]

        norm_eigenvector = (target_eigenmode - np.min(target_eigenmode)) / (
                np.max(target_eigenmode) - np.min(target_eigenmode))

        jet_cmap = plt.get_cmap('jet')

        # Convert normalized eigenvector values to RGB colors using the jet colormap
        eigenmodes_colors = (jet_cmap(norm_eigenvector)[:, :3] * 255).astype(np.uint8)

        return eigenmodes_colors

    def low_pass_filter(self,
                        k: int
                        ) -> trimesh.Trimesh:
        """
        Apply a low-pass filter to the mesh using the first k harmonics.

        Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        k (int): Number of harmonics to keep.

        Returns:
        trimesh.Trimesh: The smoothed mesh.
        """
        assert k < len(self.mesh_tri.vertices)

        U = self.graph.eig_vecs[:, :k]

        c = U.T @ self.mesh_tri.vertices
        p_smooth = U @ c

        # Update the mesh with the smoothed positions
        smoothed_mesh = self.mesh_tri.copy()
        smoothed_mesh.vertices = p_smooth

        return smoothed_mesh

    def visualize_shape_eigenmodes(self,
                                   eigenmode_indices: list[int]
                                   ) -> None:
        plotter = pv.Plotter(shape=(1, len(eigenmode_indices)))
        jet_cmap = plt.get_cmap('jet')

        for window_idx, eigenmode_idx in enumerate(eigenmode_indices):
            eigenmodes_m1_sliced = self.graph.eig_vecs[:, eigenmode_idx]
            eigenmodes_m1_n = (eigenmodes_m1_sliced - np.min(eigenmodes_m1_sliced)) / (
                    np.max(eigenmodes_m1_sliced) - np.min(eigenmodes_m1_sliced))
            clrs_m1 = (jet_cmap(eigenmodes_m1_n)[:, :3] * 255).astype(np.uint8)

            plotter.subplot(0, window_idx)
            plotter.add_text(f'Eigenmode {eigenmode_idx}', font_size=12)
            plotter.add_mesh(self.mesh_tri, scalars=clrs_m1, rgb=True, point_size=10)
            plotter.add_scalar_bar("Normalized eigenmode value", title_font_size=11, label_font_size=11)

        plotter.link_views()
        plotter.show()

if __name__ == '__main__':
    mesh = SpectralMesh('data/cow.obj', k=256)

    mesh.visualize_shape_eigenmodes([1, 2, 3, 4, 50, 100, 110])

    smoothed_mesh = mesh.low_pass_filter(k=40)
    smoothed_mesh.show()

    print("Spectral mesh processing completed.")
