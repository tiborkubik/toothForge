import math
import trimesh
import torch
import omegaconf
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree, KDTree
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


def get_data_split(cfg: omegaconf.DictConfig) -> tuple[list, list, list]:
    with open(cfg.paths.split, 'r') as file:
        lines = file.read().splitlines()

    train_ids = list()
    validation_ids = list()
    test_ids = list()

    in_train = False
    in_validation = False
    in_test = False

    for line in lines:
        if line == 'train':
            in_train = True
            in_validation = False
            in_test = False
            continue
        elif line == 'validation':
            in_train = False
            in_validation = True
            in_test = False
            continue
        elif line == 'test':
            in_train = False
            in_validation = False
            in_test = True
            continue

        if in_train:
            train_ids.append(line)
        elif in_validation:
            validation_ids.append(line)
        elif in_test:
            test_ids.append(line)

    return train_ids, validation_ids, test_ids


def custom_collate_fn(batch, apply_noise):
    case_id = [s['case_id'] for s in batch]
    path = [s['path'] for s in batch]
    R = [s['R'] for s in batch]
    c = [s['R'] @ s['c'] for s in batch]  # coefficients come already aligned from dataset's getitem
    c_original_aligned = [s['R'] @ s['c_original_aligned'] for s in batch]  # coefficients come already aligned from dataset's getitem

    for idx in range(len(c)):
        if apply_noise:
            rand_val = torch.rand(1).item()
            if rand_val < 12:
                c_augmented_low_freq = spectrum_augmenter_band(c[idx], 8, 128, 2.)
                c[idx] = c_augmented_low_freq
            else:
                ...  # 33% of cases are not modified.

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    original_sizes_v = torch.tensor([s['coords'].shape[0] for s in batch]).to(device)
    original_sizes_f = torch.tensor([s['faces'].shape[0] for s in batch]).to(device)

    padded_batch_vs = list()
    padded_batch_fs = list()
    for s in batch:
        padding = torch.zeros((max(original_sizes_v) - s['coords'].shape[0], s['coords'].shape[1]), dtype=s['coords'].dtype).to(device)
        padded_pc = torch.cat([s['coords'], padding], dim=0)
        padded_batch_vs.append(padded_pc)

        padding = torch.zeros((max(original_sizes_f) - s['faces'].shape[0], s['faces'].shape[1]), dtype=s['faces'].dtype).to(device)
        padded_fs = torch.cat([s['faces'], padding], dim=0)
        padded_batch_fs.append(padded_fs)
    return {
        'case_id': case_id,
        'path': path,
        'R': torch.stack(R).to(device),
        'c': torch.stack(c).to(device),
        'c_original_aligned': torch.stack(c_original_aligned).to(device),
        'coords_padded': torch.stack(padded_batch_vs).to(device),
        'coords_original_sizes': original_sizes_v,
        'faces_padded': torch.stack(padded_batch_fs).to(device),
        'faces_original_sizes': original_sizes_f
    }


def custom_collate_fn_augmented_batch(batch, batch_size):
    sample = batch[0]

    sample['c'] = sample['R'] @ sample['c']  # Apply synchronization on-the-fly

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    all_coeffs = torch.zeros((batch_size, sample['c'].shape[0], sample['c'].shape[1])).to(device)

    all_coeffs[0] = sample['c']
    for i in range(batch_size - 1):
        c_augmented_low_freq = spectrum_augmenter(sample['c'], 0, 0.2)
        all_coeffs[i + 1] = c_augmented_low_freq

    return {
        'R': sample['R'].unsqueeze(0).repeat(batch_size, 1, 1).to(device),
        'c': all_coeffs.to(device),
        'coords_padded': torch.tensor(sample['coords']).unsqueeze(0).repeat(batch_size, 1, 1),
        'coords_original_sizes': torch.tensor(sample['coords'].shape[0]).unsqueeze(0).repeat(batch_size),
        'faces_padded': torch.tensor(sample['faces']).unsqueeze(0).repeat(batch_size, 1, 1),
        'faces_original_sizes': torch.tensor(sample['faces'].shape[0]).unsqueeze(0).repeat(batch_size)
    }


def spectrum_augmenter(c: torch.Tensor,
                       c_id_to_modify: int,
                       noise_percentage: float = 0.1
                       ) -> torch.Tensor:
    """
    Add noise to each coefficient based on its value.

    Args:
        c (torch.Tensor): A tensor of shape (k, 3) representing a single sample's coefficients.
        c_id_to_modify (int): Index of the coefficient to augment with noise.
        noise_percentage (float): Maximum noise to add as a percentage of the coefficient value (e.g., 0.1 for 10%).

    Returns:
        torch.Tensor: The augmented coefficients with added noise.
    """
    if c_id_to_modify < 0 or c_id_to_modify >= c.shape[0]:
        raise ValueError(f"Invalid coefficient_id: {c_id_to_modify}. Must be within [0, {c.shape[0] - 1}].")

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    augmented_c = c.clone().detach()

    # Generate noise for the specified coefficient
    noise = noise_percentage * torch.abs(augmented_c[c_id_to_modify]).to(device) * (2 * torch.rand(3).to(device) - 1)

    # Apply noise to the specified coefficient
    augmented_c[c_id_to_modify] += noise

    return augmented_c


def spectrum_augmenter_band(c: torch.Tensor,
                            c_id_start: int,
                            c_id_stop: int,
                            noise_percentage: float = 0.1
                            ) -> torch.Tensor:
    augmented_c = c.clone()

    for i in range(c_id_start, c_id_stop):
        augmented_c = spectrum_augmenter(augmented_c, i, noise_percentage)

    return augmented_c


def visualize_augmented_meshes(original_mesh: trimesh.Trimesh,
                               augmented_meshes: list[trimesh.Trimesh],
                               augmentation_info: list[str]
                               ) -> None:
    plotter = pv.Plotter(shape=(1, 1 + len(augmented_meshes)))

    plotter.subplot(0, 0)
    plotter.add_text('Original Spatial Reconstruction', font_size=9)
    plotter.add_mesh(pv.wrap(original_mesh))

    for idx, augmented_mesh in enumerate(augmented_meshes):
        plotter.subplot(0, idx + 1)
        plotter.add_mesh(pv.wrap(augmented_mesh))
        plotter.add_text(augmentation_info[idx], font_size=9)

    plotter.link_views()
    plotter.show(full_screen=True)


def frange_cycle_cosine(start: float,
                        stop: float,
                        n_iter: int,
                        n_cycle: int = 6,
                        ratio: int = 0.5
                        ) -> list:
    L = np.ones(n_iter)
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    # transform into [0, pi] for plots:

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop:
            L[int(i + c * period)] = 0.5 - .5 * math.cos(v * math.pi)
            v += step
            i += 1

    L[:n_iter//n_cycle//4] = 0.0

    L[L == 1.0] = stop

    # plt.plot(L)
    # plt.show()

    return L


def calc_metrics(P: np.ndarray,
                 G: np.ndarray,
                 space_type: str) -> tuple[float, float]:
    """
    Computes the coverage metric.

    :param P: Set of synthetic point clouds of shape N x k x 3.
    :param G: Set of ground truth distribution point clouds. Shape N x k x 3.
    :param space_type: Defines whether points lie in spatial or spectral space.

    :return: value in range 0 to 1, where values close to 1 mean that gt distribution is fully covered.
    """
    assert space_type in ['spectral', 'spatial']

    coverage_arr = np.zeros(G.shape[0])  # Initially give each cloud bit value of 0.
    min_dists = np.zeros(P.shape[0])

    if space_type == 'spectral':
        # For each synthetic sample, find the closest sample in G. Mark the coverage.
        for p_idx, p in enumerate(P):
            lowest_dist = np.inf
            lowest_dist_idx = -1
            for g_idx, g in enumerate(list(G)):
                mse = np.mean((g - p) ** 2)
                if mse < lowest_dist:
                    lowest_dist = mse
                    lowest_dist_idx = g_idx
            coverage_arr[lowest_dist_idx] = 1
            min_dists[p_idx] = lowest_dist
    elif space_type == 'spatial':
        P_pcs = list(P)
        P_trees = [KDTree(p) for p in P_pcs]  # Computation of

        for p_idx, (p_pc, p_tree) in enumerate(zip(P_pcs, P_trees)):
            lowest_dist = np.inf
            lowest_dist_idx = -1
            for g_idx, gt_pc in enumerate(list(G)):
                gt_tree = KDTree(gt_pc)
                dist1, _ = p_tree.query(gt_pc)  # Closest distance from pc2 to pc1
                dist2, _ = gt_tree.query(p_pc)  # Closest distance from pc1 to pc2
                chamfer = np.mean(dist1 ** 2) + np.mean(dist2 ** 2)
                if chamfer < lowest_dist:
                    lowest_dist = chamfer
                    lowest_dist_idx = g_idx
            coverage_arr[lowest_dist_idx] = 1
            min_dists[p_idx] = lowest_dist
    else:
        raise ValueError(f'Invalid space_type: {space_type}')

    return np.sum(coverage_arr / len(coverage_arr)), np.mean(min_dists)
