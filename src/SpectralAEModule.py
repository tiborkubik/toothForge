import os
import torch
import trimesh
import wandb
import lightning as L
import torch.nn.functional as F
from utils import frange_cycle_cosine
from LearnedPoolSAE import LearnedPoolSAE
from LearnedPoolSVAE import LearnedPoolSVAE
from scipy.spatial.distance import directed_hausdorff


class SpectralAEModule(L.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 lr: float,
                 U_k_trn: torch.Tensor,
                 template_conn_trn: torch.Tensor,
                 U_k_val: torch.Tensor,
                 template_conn_val: torch.Tensor,
                 U_k_tst: torch.Tensor,
                 template_conn_tst: torch.Tensor,
                 task_name: str,
                 max_steps: int = 10,
                 kl_weight: float = -1.0,
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.lr = lr
        self.max_steps = max_steps

        self.kl_weight = kl_weight

        self.start_time = None

        self.U_k_trn = U_k_trn
        self.template_conn_trn = template_conn_trn

        self.U_k_val = U_k_val
        self.template_conn_val = template_conn_val

        self.U_k_tst = U_k_tst
        self.template_conn_tst = template_conn_tst

        self.task_name = task_name

        self.frange_cycle_cosine = frange_cycle_cosine(0, 0.2, self.max_steps, 6)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            eta_min=0.00001,
            T_max=10000,
        )

        return [optimizer], [lr_scheduler]

    @staticmethod
    def get_kl_value(log_var: torch.Tensor,
                     mu: torch.Tensor
                     ) -> torch.Tensor:
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kl_loss = kl_loss.mean(dim=0)

        return kl_loss

    def get_reconstruction_value(self,
                                 pred: torch.Tensor,
                                 gt: torch.Tensor
                                 ) -> torch.Tensor:
        if isinstance(self.model, LearnedPoolSAE):
            return F.mse_loss(pred, gt)  # Compare prediction directly, contains reconstructed coefficient
        elif isinstance(self.model, LearnedPoolSVAE):
            c_pred, _, _ = pred

            return F.mse_loss(c_pred, gt)

    def compute_loss(self, pred, c_gt):
        if isinstance(self.model, LearnedPoolSAE):
            return F.mse_loss(pred, c_gt)  # Compare prediction directly, contains reconstructed coefficient
        elif isinstance(self.model, LearnedPoolSVAE):
            c_pred, mu_pred, log_var_pred = pred

            recon_loss = F.mse_loss(c_pred, c_gt)
            # KL divergence loss
            kl_loss = self.get_kl_value(log_var_pred, mu_pred)

            if self.kl_weight == -1:
                kl_w = self.frange_cycle_cosine[self.global_step]
            else:
                kl_w = self.kl_weight

            return recon_loss + kl_w * kl_loss

    def spatial_reconstruction(self, U, pred):
        if isinstance(self.model, LearnedPoolSAE):
            return torch.matmul(U, pred)  # Compare prediction directly, contains reconstructed coefficient
        elif isinstance(self.model, LearnedPoolSVAE):
            c_pred, _, _ = pred
            return torch.matmul(U, c_pred)  # Compare prediction directly, contains reconstructed coefficient

    @staticmethod
    def spatial_reconstruction_from_c(U, c):
        return torch.matmul(U, c)  # Compare prediction directly, contains reconstructed coefficient

    def training_step(self,
                      batch,
                      batch_idx) -> dict:
        # mesh_without_augmentation = trimesh.Trimesh(vertices=(self.U_k_trn @ batch['c_original_aligned'][0]).to(torch.float16).cpu().detach().numpy(),
        #                                             faces=self.template_conn_trn.cpu().detach().numpy())
        # mesh_withaugmentation = trimesh.Trimesh(
        #     vertices=(self.U_k_trn @ batch['c'][0]).to(torch.float16).cpu().detach().numpy(),
        #     faces=self.template_conn_trn.cpu().detach().numpy())
        # 
        # mesh_without_augmentation.show()
        # mesh_withaugmentation.show()

        c_pred = self.forward(batch['c'])
        loss = self.compute_loss(c_pred, batch['c_original_aligned'])

        outputs_spatial = self.spatial_reconstruction(self.U_k_trn, c_pred)

        loss_spatial_avg = 0.0
        for i, original_size in enumerate(batch['coords_original_sizes']):
            original_pc = batch['coords_padded'][i, :original_size]  # Slice back the original point cloud
            indices1 = torch.randperm(outputs_spatial[i].shape[0])[:1024]
            indices2 = torch.randperm(original_pc.shape[0])[:1024]

            loss_spatial, _, _ = directed_hausdorff(outputs_spatial[i][indices1].to(torch.float16).cpu().detach().numpy(),
                                                    original_pc[indices2].to(torch.float16).cpu().detach().numpy())
            loss_spatial_avg += loss_spatial
        loss_spatial_avg /= batch['coords_padded'].shape[0]

        self.log('trn_loss', loss, prog_bar=True)
        self.log('trn_spectral_recon', self.get_reconstruction_value(c_pred, batch['c']), prog_bar=True)
        self.log('trn_spatial_recon', loss_spatial_avg, prog_bar=True)
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True)
        if isinstance(self.model, LearnedPoolSVAE):
            _, mu_pred, log_var_pred = c_pred
            self.log('kl', self.get_kl_value(log_var_pred, mu_pred))

            if self.kl_weight == -1:
                kl_w = self.frange_cycle_cosine[self.global_step]
            else:
                kl_w = self.kl_weight
            self.log('kl-weight', kl_w)
        return {
            'loss': loss,
            'trn_spectral_recon': self.get_reconstruction_value(c_pred, batch['c']),
            'trn_spatial_recon': loss_spatial_avg,
            'lr': self.lr_schedulers().get_last_lr()[0],
        }

    def validation_step(self, batch, batch_idx) -> dict:
        c_pred = self.forward(batch['c'])

        loss = self.compute_loss(c_pred, batch['c_original_aligned'])  # Here it could be either c or c_orig_al, since in validation to noise is added.

        outputs_spatial = self.spatial_reconstruction(self.U_k_val, c_pred)

        loss_spatial_avg = 0.0
        for i, original_size in enumerate(batch['coords_original_sizes']):
            if (i == 0 and self.current_epoch % 7500 == 0) or (self.current_epoch % 2500 == 0):
                mesh_tri = trimesh.Trimesh(vertices=torch.matmul(self.U_k_val, batch['c'])[i].to(torch.float16).detach().cpu().numpy(),
                                           faces=self.template_conn_val.cpu().detach().numpy())
                mesh_tri.export(os.path.join('logs/media', self.task_name, f'{self.current_epoch}-{i}-gt_val.stl'))

                mesh_tri = trimesh.Trimesh(vertices=outputs_spatial[i].to(torch.float16).cpu().detach().numpy(),
                                           faces=self.template_conn_val.cpu().detach().numpy())
                mesh_tri.export(os.path.join('logs/media', self.task_name, f'{self.current_epoch}-{i}-pred_val.stl'))

                wandb.log({f'Reconstruction {i} GT': wandb.Object3D(open(os.path.join('logs/media', self.task_name, f'{self.current_epoch}-{i}-gt_val.stl'))),
                           f'Reconstruction {i} Pred': wandb.Object3D(open(os.path.join('logs/media', self.task_name, f'{self.current_epoch}-{i}-pred_val.stl')))})

            original_pc = batch['coords_padded'][i, :original_size]  # Slice back the original point cloud
            indices1 = torch.randperm(outputs_spatial[i].shape[0])[:1024]
            indices2 = torch.randperm(original_pc.shape[0])[:1024]

            loss_spatial, _, _ = directed_hausdorff(outputs_spatial[i][indices1].to(torch.float16).cpu().detach().numpy(),
                                                    original_pc[indices2].to(torch.float16).cpu().detach().numpy())
            loss_spatial_avg += loss_spatial
        loss_spatial_avg /= batch['coords_padded'].shape[0]

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_spectral_recon', self.get_reconstruction_value(c_pred, batch['c']))
        self.log('val_spatial_recon', loss_spatial_avg, prog_bar=True)
        if isinstance(self.model, LearnedPoolSVAE):
            _, mu_pred, log_var_pred = c_pred
            self.log('kl', self.get_kl_value(log_var_pred, mu_pred))
        return {
            'val_loss': loss,
            'val_spectral_recon': self.get_reconstruction_value(c_pred, batch['c']),
            'val_spatial_recon': loss_spatial_avg,
        }
