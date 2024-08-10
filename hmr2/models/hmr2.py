import torch
import pytorch_lightning as pl
from typing import Dict, Tuple, Any

from yacs.config import CfgNode
from ..utils import SkeletonRenderer, MeshRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads import build_smpl_head
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from . import SMPL

log = get_pylogger(__name__)

class HMR2(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
            self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict'])

        self.smpl_head = build_smpl_head(cfg)

        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.discriminator = Discriminator()

        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.smpl_parameter_loss = ParameterLoss()

        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)

        self.register_buffer('initialized', torch.tensor(False))

        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.smpl.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

        self.automatic_optimization = False

    def get_parameters(self):
        all_params = list(self.smpl_head.parameters())
        all_params += list(self.backbone.parameters())
        return all_params

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]
        optimizer = torch.optim.AdamW(params=param_groups, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(), lr=self.cfg.TRAIN.LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        return optimizer, optimizer_disc

    def forward_step(self, x: torch.Tensor, train: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        conditioning_feats = self.backbone(x[:, :, :, 32:-32])
        pred_smpl_params, pred_cam, _ = self.smpl_head(conditioning_feats)

        output = {}
        output['pred_cam'] = pred_cam
        output['pred_smpl_params'] = {k: v.clone() for k, v in pred_smpl_params.items()}

        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9)], dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)
        smpl_output = self.smpl(**{k: v.float() for k, v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        return output

    def compute_loss(self, batch: Dict[str, torch.Tensor], output: Dict[str, torch.Tensor], train: bool = True) -> torch.Tensor:
        pred_smpl_params = output['pred_smpl_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']

        batch_size = pred_smpl_params['body_pose'].shape[0]
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype

        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_smpl_params = batch['smpl_params']
        has_smpl_params = batch['has_smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']

        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=25+14)

        loss_smpl_params = {}
        for k, pred in pred_smpl_params.items():
            gt = gt_smpl_params[k].view(batch_size, -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_smpl_params[k]
            loss_smpl_params[k] = self.smpl_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt)

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d +\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d +\
               sum([loss_smpl_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_smpl_params])

        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach())

        for k, v in loss_smpl_params.items():
            losses['loss_' + k] = v.detach()

        output['losses'] = losses

        return loss

    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict[str, torch.Tensor], output: Dict[str, torch.Tensor], step_count: int, train: bool = True, write_to_summary_writer: bool = True) -> None:
        mode = 'train' if train else 'val'
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)

        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)
        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        gt_keypoints_3d = batch['keypoints_3d']
        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)

        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode + '/' + loss_name, val.detach().item(), step_count)
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        gt_keypoints_3d = batch['keypoints_3d']
        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)

        predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                               pred_cam_t[:num_images].cpu().numpy(),
                                                               images[:num_images].cpu().numpy(),
                                                               pred_keypoints_2d[:num_images].cpu().numpy(),
                                                               gt_keypoints_2d[:num_images].cpu().numpy(),
                                                               focal_length=focal_length[:num_images].cpu().numpy())
        if write_to_summary_writer:
            summary_writer.add_image('%s/predictions' % mode, predictions, step_count)

        return predictions

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward_step(x, train=False)

    def training_step_discriminator(self, batch: Dict[str, torch.Tensor], body_pose: torch.Tensor, betas: torch.Tensor, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        batch_size = body_pose.shape[0]
        gt_body_pose = batch['body_pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_body_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(body_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def training_step(self, joint_batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        batch = joint_batch['img']
        mocap_batch = joint_batch['mocap']
        optimizer = self.optimizers(use_pl_optimizer=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer, optimizer_disc = optimizer

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_smpl_params = output['pred_smpl_params']
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output, train=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            disc_out = self.discriminator(pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv

        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        optimizer.step()
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            loss_disc = self.training_step_discriminator(mocap_batch, pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1), optimizer_disc)
            output['losses']['loss_gen'] = loss_adv
            output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return output

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx=0) -> Dict[str, Any]:
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        self.tensorboard_logging(batch, output, self.global_step, train=False)
        return output

    def export_to_onnx(self, dummy_input: torch.Tensor, file_path: str) -> None:
        torch_input = torch.randn(1, 3, 256, 256)
        onnx_program = torch.onnx.dynamo_export(self, torch_input)
    def export_to_onnx2(self, dummy_input: torch.Tensor, file_path: str) -> None:
        torch.onnx.export(self,               # model being run
        torch.randn((1, 3, 224, 224)),         # model input (or a tuple for multiple inputs)
        file_path,           # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,    # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],   # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                      'output': {0: 'batch_size'}})  
