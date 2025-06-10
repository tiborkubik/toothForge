import os
import trimesh
import argparse
import numpy as np
from copy import deepcopy

from src.SpectralMesh import SpectralMesh
from src.utils import find_rotation_closed_form_iterative

_K_LIMIT: int = 512

_RUN_PREALIGNMENT: bool = True

_DEFAULT_MESH_DATASET_PATH: str = '../data/meshes/'
_DEFAULT_SPEC_DATASET_PATH: str = '../data/generated/'

_PERMUTATIONS = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
_FLIPS = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
          [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]]


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--folder-path-in', type=str, default=_DEFAULT_MESH_DATASET_PATH,
                      help='Path to dataset folder. This folder contains files in .stl/.obj format.')
    args.add_argument('--folder-path-out', type=str, default=_DEFAULT_SPEC_DATASET_PATH,
                      help='Path to the folder where outputs should be generated.')

    args.add_argument('--k', type=str, default=_K_LIMIT,
                      help=f'How many eigenvectors to generate. Default is {_K_LIMIT}.')

    args = args.parse_args()

    return args


def get_alignment(target_mesh_path: str,
                  case_id: str,
                  mesh_template_spec: SpectralMesh,
                  args,
                  ):
    print(f'Aligning case {case_id}...')
    mesh_target_spec = SpectralMesh(os.path.join(args.folder_path_in, target_mesh_path), k=args.k)

    c_template = np.asarray(mesh_template_spec.graph.eig_vecs.T @ mesh_template_spec.mesh_tri.vertices)
    U_common = mesh_template_spec.graph.eig_vecs

    alignment_opts = {
        'niter': [50, 50, 20, 10, 10],
        'kr': [
            np.arange(4),
            np.arange(6),
            np.arange(8),
            np.arange(20),
            np.arange(args.k),
        ],
    }

    if _RUN_PREALIGNMENT:
        avg_volume_ins = (mesh_template_spec.mesh_tri.volume + mesh_target_spec.mesh_tri.volume) / 2.

        lowest_err = float('inf')
        best_perm = [0, 1, 2]  # Defaults
        best_flip = [1, 1, 1]  # Defaults
        for perm in _PERMUTATIONS:
            for flip in _FLIPS:
                m_g_work_copy = deepcopy(mesh_target_spec)
                m_g_work_copy.graph.X[:, :3] = m_g_work_copy.graph.X[:, perm]
                m_g_work_copy.graph.eig_vecs[:, :3] = m_g_work_copy.graph.eig_vecs[:, perm]
                m_g_work_copy.graph.eig_vecs_inv[:3, :] = m_g_work_copy.graph.eig_vecs_inv[perm, :]

                for eigenmode_id in [0, 1, 2]:
                    m_g_work_copy.graph.X[:, eigenmode_id] *= flip[eigenmode_id]
                    m_g_work_copy.graph.eig_vecs[:, eigenmode_id] *= flip[eigenmode_id]
                    m_g_work_copy.graph.eig_vecs_inv[eigenmode_id, :] *= flip[eigenmode_id]

                corr = find_rotation_closed_form_iterative(mesh_template_spec, m_g_work_copy, alignment_opts)

                w_f = mesh_template_spec.graph.eig_vecs.T @ mesh_template_spec.mesh_tri.vertices  # Shape of w_f: [k, 3]
                w_g = m_g_work_copy.graph.eig_vecs.T @ m_g_work_copy.mesh_tri.vertices  # Shape of w_g: [k, 3]
                U_common = mesh_template_spec.graph.eig_vecs

                try:
                    temp_w_g = corr['R_21'] @ w_g
                except ValueError:
                    continue

                w_p = .5 * w_f + .5 * temp_w_g  # shape of w_p: [k, 3]
                pos_p = U_common @ w_p  # Reconstruction, shape: [N, 3]

                mesh_tri = trimesh.Trimesh(vertices=pos_p, faces=mesh_template_spec.mesh_tri.faces)
                mid_shape_volume = np.abs(mesh_tri.volume)
                diff = np.abs(mid_shape_volume - avg_volume_ins)
                if diff < lowest_err:
                    lowest_err = diff
                    best_perm = perm
                    best_flip = flip
                    print(f'Best-performing combination: permutation: {perm}, flip: {flip}')

        mesh_target_spec.graph.X[:, :3] = mesh_target_spec.graph.X[:, best_perm]
        mesh_target_spec.graph.eig_vecs[:, :3] = mesh_target_spec.graph.eig_vecs[:, best_perm]
        mesh_target_spec.graph.eig_vecs_inv[:3, :] = mesh_target_spec.graph.eig_vecs_inv[best_perm, :]

        for eigenmode_id in [0, 1, 2]:
            mesh_target_spec.graph.X[:, eigenmode_id] *= best_flip[eigenmode_id]
            mesh_target_spec.graph.eig_vecs[:, eigenmode_id] *= best_flip[eigenmode_id]
            mesh_target_spec.graph.eig_vecs_inv[eigenmode_id, :] *= best_flip[eigenmode_id]

    corr = find_rotation_closed_form_iterative(mesh_template_spec, mesh_target_spec, alignment_opts)

    # Store reconstructions as well.
    c_unaligned = np.asarray(mesh_target_spec.graph.eig_vecs.T @ mesh_target_spec.mesh_tri.vertices)
    c_aligned = corr['R_21'] @ c_unaligned
    c_intp = .5 * c_template + .5 * c_aligned

    pos_intp = U_common @ c_intp
    mesh_tri_intp = trimesh.Trimesh(vertices=pos_intp, faces=mesh_template_spec.mesh_tri.faces)

    pos_unaligned = U_common @ c_unaligned
    mesh_tri_unaligned = trimesh.Trimesh(vertices=pos_unaligned, faces=mesh_template_spec.mesh_tri.faces)

    pos_aligned = U_common @ c_aligned
    mesh_tri_aligned = trimesh.Trimesh(vertices=pos_aligned, faces=mesh_template_spec.mesh_tri.faces)

    print(f'Case {case_id} successfully aligned.')

    return mesh_target_spec, corr, mesh_tri_intp, mesh_tri_unaligned, mesh_tri_aligned, case_id


def main() -> None:
    args = parse_args()

    os.makedirs(args.folder_path_out, exist_ok=True)  # Create empty output folder if it does not exist yet.

    mesh_ps = [f for f in os.listdir(args.folder_path_in) if f.endswith('.obj') or f.endswith('.stl')]

    ''' Processing template mesh. '''
    mesh_template_p = mesh_ps[0]  # First mesh will be set as the template.
    mesh_template_spec = SpectralMesh(os.path.join(args.folder_path_in, mesh_template_p), k=args.k)

    template_f_p = os.path.join(args.folder_path_out, 'template')
    os.makedirs(template_f_p, exist_ok=True)
    mesh_template_spec.store_data(template_f_p, is_template=True)

    target_meshes_paths = mesh_ps[1:]
    target_ids = [str(i).zfill(6) for i in range(len(target_meshes_paths))]

    ''' Processing other samples. '''
    for sample, target_id in zip(target_meshes_paths, target_ids):
        mesh_target_spec, corr, mesh_tri_intp, mesh_tri_unaligned, mesh_tri_aligned, _ = get_alignment(sample,
                                                                                                       target_id,
                                                                                                       mesh_template_spec,
                                                                                                       args)

        mesh_target_spec.store_data(os.path.join(args.folder_path_out, target_id), R=corr['R_21'])

        mesh_tri_intp.export(os.path.join(args.folder_path_out, target_id, 'reconstructed_interp.stl'))
        mesh_tri_unaligned.export(os.path.join(args.folder_path_out, target_id, 'reconstructed_unaligned.stl'))
        mesh_tri_aligned.export(os.path.join(args.folder_path_out, target_id, 'reconstructed_aligned.stl'))


if __name__ == '__main__':
    main()
