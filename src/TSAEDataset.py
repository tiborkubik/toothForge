import os
import h5py
import torch
import trimesh
import pyvista as pv

from torch.utils.data import Dataset
from utils import spectrum_augmenter, spectrum_augmenter_band, visualize_augmented_meshes


class TSAEDataset(Dataset):
    def __init__(self,
                 path: str,
                 device: str,
                 k: int,
                 ids: list,
                 transformation_num: int = 0,
                 apply_transform: bool = False,
                 ) -> None:
        self.path = path
        self.device = device
        self.k = k
        self.ids = ids
        self.apply_transform = apply_transform
        self.transformation_num = transformation_num

        self.case_paths, self.case_ids = zip(*(
        [(f.path, f.name) for f in os.scandir(self.path) if f.is_dir() and f.name in self.ids or f.name == 'template']))

        assert f'template' in self.case_ids, f'Path {self.path} does not contain template shape.'

        # Preparing eigenvectors for projections, based on the template shape.
        self.l, self.U = self.load_common_eigen_space()
        assert self.k <= self.l.shape[0], (f'Expected k value ({self.k}) is larger '
                                           f'than generated spectrum ({self.l.shape[0]}). Aborting.')
        self.l_k, self.U_k = self.l[:self.k], self.U[:, :self.k]

        # Preparing template shape geometry for reconstructions.
        self.templ_coords, self.templ_faces = self.load_template_geometry()

        self.samples: list[dict] = self.load_samples()  # Loads a list of dicts of all samples, also containing template.

        self.noise_percentage = 4.

    def load_common_eigen_space(self) -> tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(os.path.join(self.path, 'template', 'spectrum.h5'), 'r') as h5_file:
            l = h5_file['evals'][:]  # Eigen values (lambdas)
            U = h5_file['evecs'][:]

            l = torch.tensor(l, dtype=torch.float32, device=self.device)
            U = torch.tensor(U, dtype=torch.float32, device=self.device)

        return l, U

    def load_template_geometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(os.path.join(self.path, 'template', 'spatial.h5'), 'r') as h5_file:
            coords = h5_file['coords'][:]  # Eigen values (lambdas)
            faces = h5_file['faces'][:]

            coords = torch.tensor(coords, dtype=torch.float32, device=self.device)
            faces = torch.tensor(faces, dtype=torch.int32, device=self.device)

        return coords, faces

    def load_samples(self) -> list:
        samples = list()

        for case_id, case_p in zip(self.case_ids, self.case_paths):
            case_dict = dict()
            print(case_id)
            case_dict['case_id'] = case_id
            case_dict['path'] = case_p

            with h5py.File(os.path.join(self.path, case_id, 'spec_coeff.h5'), 'r') as h5_file:
                c = h5_file['c'][:]  # Spectral coefficients
                c = torch.tensor(c, dtype=torch.float32, device=self.device)

                case_dict['c'] = c

            with h5py.File(os.path.join(self.path, case_id, 'spatial.h5'), 'r') as h5_file:
                coords = h5_file['coords'][:]
                coords = torch.tensor(coords, dtype=torch.float32, device=self.device)
                case_dict['coords'] = coords

                faces = h5_file['faces'][:]
                faces = torch.tensor(faces, dtype=torch.int16, device=self.device)
                case_dict['faces'] = faces

            if case_id != 'template':  # Special case: template does not have spec_rotation available (its identity).
                with h5py.File(os.path.join(self.path, case_id, 'spec_rotation.h5'), 'r') as h5_file:
                    R = h5_file['R'][:]  # Rotation matrices
                    R = torch.tensor(R, dtype=torch.float32, device=self.device)

                    case_dict['R'] = R
            else:
                case_dict['R'] = torch.eye(self.l.shape[0], dtype=torch.float32, device=self.device)

            samples.append(case_dict)
            # # # TODO: temporary code location.
            # c_aligned = torch.matmul(case_dict['R'], case_dict['c'])
            #
            # coords_spatial = torch.matmul(self.U, c_aligned)
            # mesh_tri = trimesh.Trimesh(vertices=coords_spatial.detach().cpu().numpy(),
            #                            faces=self.templ_faces.cpu().detach().numpy())
            #
            # augmented_meshes: list[trimesh.Trimesh] = list()
            # aug_info: list[str] = list()
            # for i in range(10):
            #     noise_percentage = 5.
            #
            #     # c_augmented_low_freq = spectrum_augmenter(c_aligned, 1, noise_percentage)
            #     # c_augmented_low_freq = spectrum_augmenter(c_aligned, 1, 0.4)
            #     # c_augmented_low_freq = spectrum_augmenter(c_aligned, 2, .6)
            #     c_augmented_low_freq = spectrum_augmenter_band(c_aligned, 16, 64, noise_percentage)
            #
            #     coords_spatial_aug = torch.matmul(self.U, c_augmented_low_freq)
            #     mesh_tri_aug = trimesh.Trimesh(vertices=coords_spatial_aug.detach().cpu().numpy(),
            #                                    faces=self.templ_faces.cpu().detach().numpy())
            #     augmented_meshes.append(mesh_tri_aug)
            #     mesh_tri_aug.export(f'../data/pcs-for-fid/incisors/distorted-{i}-{case_id}.stl')
            #     points, _ = trimesh.sample.sample_surface(mesh_tri_aug, 16000)
            #     pc = trimesh.PointCloud(points)
            #     pc.export(f'../data/pcs-for-fid/incisors/distorted-{i}-{case_id}.ply')
            #
            #     aug_info.append(f'Aug. coeff: 16-32, max_noise_percentage: {noise_percentage}')
            #
            # visualize_augmented_meshes(mesh_tri, augmented_meshes, aug_info)

            ...

        return samples

    def __getitem__(self, index) -> dict:
        """

        :param index: id of sample from given dataset
        :return: Spectral coefficient representation of given shape. Coefficients are spectrally aligned and if asked,
        also augmented with probability of 60%.
        """
        case_dict = self.samples[index]

        c_aligned = case_dict['c']
        case_dict['c_original_aligned'] = c_aligned  # Without augmentation

        return case_dict

    def __len__(self):
        return len(self.samples)
