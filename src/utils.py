import trimesh
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

from src.SpectralMesh import SpectralMesh


def show_matrix_subpart(array: np.ndarray, size_to_show: int = 10) -> None:
    # Slice the MxM subarray from the top-left corner
    subarray = array[:size_to_show, :size_to_show]

    # Visualize the MxM subarray with Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(subarray, cmap='viridis', interpolation='none')
    plt.colorbar(label='Value')

    # Add annotations for each cell
    for i in range(size_to_show):
        for j in range(size_to_show):
            value = subarray[i, j]
            plt.text(j, i, f'{value:.2f}', ha='center', va='center', color='white')

    plt.title(f'{size_to_show}x{size_to_show} Slice of {array.shape[0]}x{array.shape[1]} Array')
    plt.show()


def chamfer_distance(mesh1: SpectralMesh | trimesh.Trimesh,
                     mesh2: SpectralMesh | trimesh.Trimesh
                     ) -> tuple[float, np.ndarray]:
    """
    Compute the Chamfer distance between two meshes and return the point-wise errors.
    """

    if isinstance(mesh1, SpectralMesh):
        vs1 = mesh1.mesh_tri.vertices
    elif isinstance(mesh1, trimesh.Trimesh):
        vs1 = mesh1.vertices
    else:
        raise TypeError(f'Parameter mesh1 is of invalid type.')

    if isinstance(mesh2, SpectralMesh):
        vs2 = mesh2.mesh_tri.vertices
    elif isinstance(mesh1, trimesh.Trimesh):
        vs2 = mesh2.vertices
    else:
        raise TypeError(f'Parameter mesh1 is of invalid type.')

    tree1 = cKDTree(vs1)
    tree2 = cKDTree(vs2)

    dists_1_to_2, _ = tree1.query(vs2, k=1)
    dists_2_to_1, _ = tree2.query(vs1, k=1)

    # Chamfer distance
    chamfer_dist = np.mean(dists_1_to_2 ** 2) + np.mean(dists_2_to_1 ** 2)

    # Return Chamfer distance and point-wise distances for visualization
    return chamfer_dist, dists_1_to_2


def find_closest_points_in_spectral_domain(X_1, X_2):
    tree_X_2 = cKDTree(X_2)
    _, corr_1_2 = tree_X_2.query(X_1)  # all corresponding points from embedding 1 onto embedding 2

    del tree_X_2

    return corr_1_2


def find_rotation_closed_form_iterative(m1: SpectralMesh,
                                        m2: SpectralMesh,
                                        opts: dict,
                                        data_term: bool = True,
                                        ) -> dict:
    Z_1 = m1.graph.X[:, :3]
    Z_2 = m2.graph.X[:, :3]

    corr: dict = {
        'corr_12': None,
        'corr_21': None,
        'C_12': None,
        'C_21': None,
        'R_12': None,
        'R_21': None
    }

    if data_term:
        assert m1.extended_data is not None
        assert m2.extended_data is not None
        Z_1 = np.concatenate((Z_1, m1.extended_data), axis=1)
        Z_2 = np.concatenate((Z_2, m2.extended_data), axis=1)

    Z_1_o = Z_1
    Z_2_o = Z_2

    last_err_Z = 1e10
    last_Z_1 = Z_1
    last_Z_2 = Z_2

    errs_X = list()
    errs_Z = list()
    print(f'Align Embeddings')

    for iter_recon in range(len(opts['niter'])):
        # Perform the inner loop using the value in the list as the number of iterations
        for iter in range(opts['niter'][iter_recon]):
            corr['corr_12'] = find_closest_points_in_spectral_domain(Z_1_o,
                                                                     Z_2)  # closest points of M1 in M2. Shape: [n_z_1, ]
            corr['corr_21'] = find_closest_points_in_spectral_domain(Z_2_o, Z_1)

            C_12 = coo_matrix((np.ones(Z_1_o.shape[0]),
                               (np.arange(Z_1_o.shape[0]), corr['corr_12'])),
                              shape=(Z_1_o.shape[0], Z_2_o.shape[0]))
            C_21 = coo_matrix((np.ones(Z_2_o.shape[0]),
                               (np.arange(Z_2_o.shape[0]), corr['corr_21'])),
                              shape=(Z_2_o.shape[0], Z_1_o.shape[0]))

            err_X = np.sum(np.sum((Z_1[corr['corr_21'], :3] - Z_2_o[:, :3]) ** 2, axis=1))
            err_Z = np.sum(np.sum((Z_1[corr['corr_21'], :] - Z_2_o) ** 2, axis=1))
            errs_X.append(err_X)
            errs_Z.append(err_Z)
            print(f'[{iter_recon} - {iter}/{opts["niter"][iter_recon]}, {opts["kr"][iter_recon][-1] + 1} eigenmodes] '
                  f'Total sum of squared differences X: {err_X}'
                  f'Total sum of squared differences Z: {err_Z}')

            if err_Z > last_err_Z:
                Z_1 = last_Z_1
                Z_2 = last_Z_2
                break

            last_err_Z = err_Z
            last_Z_1 = Z_1
            last_Z_2 = Z_2

            # Direction 1
            R_21 = m1.graph.eig_vecs_inv[opts['kr'][iter_recon], :] @ C_12 @ m2.graph.eig_vecs_inv[
                                                                             opts['kr'][iter_recon], :].T
            w_1 = m1.graph.eig_vecs_inv[opts['kr'][iter_recon], :] @ m1.graph.X[:, opts['kr'][iter_recon]]
            w_2 = m2.graph.eig_vecs_inv[opts['kr'][iter_recon], :] @ m2.graph.X[:, opts['kr'][iter_recon]]
            w = R_21 @ w_2

            Y_1 = m1.graph.eig_vecs[:, opts['kr'][iter_recon]] @ w
            Y_1 = Y_1[:, :3]

            if data_term:
                Z_1 = Y_1
                Z_1 = np.concatenate((Z_1, m1.extended_data), axis=1)
            else:
                Z_1 = Y_1

            # Direction 2
            R_12 = m2.graph.eig_vecs_inv[opts['kr'][iter_recon], :] @ C_21 @ m1.graph.eig_vecs_inv[
                                                                             opts['kr'][iter_recon], :].T
            w = R_12 @ w_1
            Y_2 = m2.graph.eig_vecs[:, opts['kr'][iter_recon]] @ w

            Y_2 = Y_2[:, :3]

            if data_term:
                Z_2 = Y_2
                Z_2 = np.concatenate((Z_2, m2.extended_data), axis=1)
            else:
                Z_2 = Y_2

    corr['C_12'] = C_12
    corr['C_21'] = C_21
    corr['R_12'] = R_12
    corr['R_21'] = R_21

    return corr
