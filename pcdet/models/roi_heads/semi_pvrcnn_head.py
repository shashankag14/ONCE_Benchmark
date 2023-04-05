from pcdet.models.roi_heads.pvrcnn_head import PVRCNNHead
from pcdet.models.roi_heads.target_assigner.proposal_target_layer_ulb import UnlabeledProposalTargetLayer
import numpy as np
import torch
from ...utils import common_utils
import torch.nn.functional as F

class SemiPVRCNNHead(PVRCNNHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.proposal_target_layer_ulb = UnlabeledProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.model_type = kwargs.get('model_type')

    def forward_ulb(self, batch_dict):

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            # For analysis
            targets_dict['gt_boxes'] = batch_dict['gt_boxes']
            targets_dict['pl_scores'] = batch_dict['pl_scores']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)


        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
        )
        batch_dict['batch_cls_preds'] = batch_cls_preds
        batch_dict['batch_box_preds'] = batch_box_preds
        batch_dict['cls_preds_normalized'] = False

        if self.training:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            # Temporarily add to targets_dict for metrics
            targets_dict['batch_box_preds'] = batch_box_preds

            self.forward_ret_dict = targets_dict

        return batch_dict

    def generate_proposals(self, batch_dict):
        return self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

    def forward(self, batch_dict):
        if batch_dict['type'] == 'unlabeled' and self.model_type == 'student':
            return self.forward_ulb(batch_dict)
        else:
            return super().forward(batch_dict)

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            if batch_dict['type'] == 'unlabeled':
                targets_dict = self.proposal_target_layer_ulb.forward(batch_dict)
            else:
                targets_dict = self.proposal_target_layer.forward(batch_dict)

        rois = targets_dict['rois']  # (B, N, 7 + C)
        #print("After proposal target layer")
        #print(rois.shape)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)

        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            batch_size = forward_ret_dict['rcnn_cls_labels'].shape[0]
            batch_loss_cls = batch_loss_cls.reshape(batch_size, -1)
            cls_valid_mask = cls_valid_mask.reshape(batch_size, -1)
            if 'rcnn_cls_weights' in forward_ret_dict:
                weights = forward_ret_dict['rcnn_cls_weights'].view(batch_size, -1)
                rcnn_loss_cls_norm = (cls_valid_mask * weights).sum(-1)
                rcnn_loss_cls = (batch_loss_cls * cls_valid_mask * weights).sum(-1) / torch.clamp(rcnn_loss_cls_norm, min=1.0)
            else:
                rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss_ulb(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        enable_st = self.model_cfg.get('ENABLE_SOFT_TEACHER', False)
        enable_ulb_obj = self.model_cfg.get('UNLABELED_SUPERVISE_OBJ', False)
        if enable_st or enable_ulb_obj:
            rcnn_cls_loss, rcnn_cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
            rcnn_loss += rcnn_cls_loss
            tb_dict.update(rcnn_cls_tb_dict)
        if self.model_cfg.get('UNLABELED_SUPERVISE_REFINE', False):
            rcnn_reg_loss, rcnn_reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
            rcnn_loss += rcnn_reg_loss
            tb_dict.update(rcnn_reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss

        return rcnn_loss, tb_dict