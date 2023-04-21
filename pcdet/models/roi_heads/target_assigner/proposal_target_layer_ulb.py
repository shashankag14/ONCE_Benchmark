import numpy as np
import torch
import torch.nn as nn
from pcdet.ops.iou3d_nms import iou3d_nms_utils

from pcdet.models.roi_heads.target_assigner.proposal_target_layer import ProposalTargetLayer

class UnlabeledProposalTargetLayer(ProposalTargetLayer):
    def forward(self, batch_dict):
        return self.sample_rois_for_rcnn(batch_dict=batch_dict)

    def sample_rois_for_rcnn(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1)
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        # batch_gt_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_reg_valid_mask = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)
        batch_cls_labels = -rois.new_ones(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        interval_mask = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, dtype=torch.bool)
        batch_rcnn_cls_weights = torch.ones_like(batch_cls_labels)

        for index in range(batch_size):
            cur_gt_boxes = batch_dict['gt_boxes'][index]
            k = cur_gt_boxes.__len__() - 1
            while k >= 0 and cur_gt_boxes[k].sum() == 0:
                k -= 1
            cur_gt_boxes = cur_gt_boxes[:k + 1]
            cur_gt_boxes = cur_gt_boxes.new_zeros((1, cur_gt_boxes.shape[1])) if len(
                cur_gt_boxes) == 0 else cur_gt_boxes

            cur_rois = batch_dict['rois'][index]
            cur_roi_labels = batch_dict['roi_labels'][index]

            subsample_unlabeled_rois = getattr(self, self.roi_sampler_cfg.UNLABELED_SAMPLER_TYPE, None)
            sampled_inds, cur_reg_valid_mask, cur_cls_labels, roi_ious, gt_assignment, cur_interval_mask = subsample_unlabeled_rois(
                cur_rois, cur_gt_boxes, cur_roi_labels)

            cur_roi = batch_dict['rois'][index][sampled_inds]
            cur_roi_scores = batch_dict['roi_scores'][index][sampled_inds]
            cur_roi_labels = batch_dict['roi_labels'][index][sampled_inds]
            batch_roi_ious[index] = roi_ious
            # batch_gt_scores[index] = batch_dict['pred_scores_ema'][index][sampled_inds]
            batch_gt_of_rois[index] = cur_gt_boxes[gt_assignment[sampled_inds]]

            batch_rois[index] = cur_roi
            batch_roi_labels[index] = cur_roi_labels
            batch_roi_scores[index] = cur_roi_scores
            interval_mask[index] = cur_interval_mask
            batch_reg_valid_mask[index] = cur_reg_valid_mask
            batch_cls_labels[index] = cur_cls_labels

        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': batch_reg_valid_mask, 'rcnn_cls_labels': batch_cls_labels,
                        'interval_mask': interval_mask, 'rcnn_cls_weights': batch_rcnn_cls_weights}

        return targets_dict

    def subsample_rois_topk(self, rois, gt_boxes, roi_labels):
        if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
            max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                rois=rois, roi_labels=roi_labels,
                gt_boxes=gt_boxes[:, 0:7], gt_labels=gt_boxes[:, -1].long()
            )
        else:
            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(rois, gt_boxes[:, 0:7])  # (M, N)
            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        bg_rois_per_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_image
        if self.roi_sampler_cfg.get('UNLABELED_SAMPLE_EASY_BG', False):
            hard_bg_rois_per_image = (int(bg_rois_per_image * self.roi_sampler_cfg.HARD_BG_RATIO))
        else:
            hard_bg_rois_per_image = bg_rois_per_image

        _, sampled_inds = torch.topk(max_overlaps, k=fg_rois_per_image + hard_bg_rois_per_image)

        if self.roi_sampler_cfg.get('UNLABELED_SAMPLE_EASY_BG', False):
            easy_bg_rois_per_image = bg_rois_per_image - hard_bg_rois_per_image
            _, easy_bg_inds = torch.topk(max_overlaps, k=easy_bg_rois_per_image, largest=False)
            sampled_inds = torch.cat([sampled_inds, easy_bg_inds])

        roi_ious = max_overlaps[sampled_inds]
        gt_of_rois = gt_boxes[gt_assignment[sampled_inds]]

        rcnn_cls_labels, reg_valid_mask, interval_mask = self.classwise_iou_filtering(gt_of_rois, roi_ious, roi_labels)

        # interval_mask, reg_valid_mask and cls_labels are defined in pre_loss_filtering based on advanced thresholding.
        # cur_reg_valid_mask = torch.zeros_like(sampled_inds, dtype=torch.int)
        # cur_cls_labels = -torch.ones_like(sampled_inds, dtype=torch.float)
        # interval_mask = torch.zeros_like(sampled_inds, dtype=torch.bool)

        return sampled_inds, reg_valid_mask, rcnn_cls_labels, roi_ious, gt_assignment, interval_mask

    def subsample_rois_classwise(self, rois, gt_boxes, roi_labels):
        reg_fg_thresh = rois.new_tensor(self.roi_sampler_cfg.UNLABELED_REG_FG_THRESH).view(1, -1).repeat(len(rois), 1)
        reg_fg_thresh = reg_fg_thresh.gather(dim=-1, index=(roi_labels - 1).unsqueeze(-1)).squeeze(-1)

        cls_fg_thresh = rois.new_tensor(self.roi_sampler_cfg.UNLABELED_CLS_FG_THRESH).view(1, -1).repeat(len(rois), 1)
        cls_fg_thresh = cls_fg_thresh.gather(dim=-1, index=(roi_labels - 1).unsqueeze(-1)).squeeze(-1)

        if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
            max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                rois=rois, roi_labels=roi_labels,
                gt_boxes=gt_boxes[:, 0:7], gt_labels=gt_boxes[:, -1].long()
            )
        else:
            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(rois, gt_boxes[:, 0:7])  # (M, N)
            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

        sampled_inds = self.subsample_rois(max_overlaps=max_overlaps,
                                           reg_fg_thresh=reg_fg_thresh,
                                           cls_fg_thresh=cls_fg_thresh)
        roi_ious = max_overlaps[sampled_inds]

        if isinstance(reg_fg_thresh, torch.Tensor):
            reg_fg_thresh = reg_fg_thresh[sampled_inds]
        if isinstance(cls_fg_thresh, torch.Tensor):
            cls_fg_thresh = cls_fg_thresh[sampled_inds]

        # regression valid mask
        reg_valid_mask = (roi_ious > reg_fg_thresh).long()

        # classification label
        iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH

        iou_fg_thresh = cls_fg_thresh

        fg_mask = roi_ious > iou_fg_thresh
        bg_mask = roi_ious < iou_bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)
        cls_labels = (fg_mask > 0).float()
        cls_labels[interval_mask] = (roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh[interval_mask] - iou_bg_thresh)

        ignore_mask = torch.eq(gt_boxes[gt_assignment[sampled_inds]], 0).all(dim=-1)
        cls_labels[ignore_mask] = -1

        return sampled_inds, reg_valid_mask, cls_labels, roi_ious, gt_assignment, interval_mask

    def subsample_rois(self, max_overlaps, reg_fg_thresh=None, cls_fg_thresh=None):
        if reg_fg_thresh is None:
            reg_fg_thresh = self.roi_sampler_cfg.REG_FG_THRESH
        if cls_fg_thresh is None:
            cls_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        if isinstance(reg_fg_thresh, torch.Tensor):
            fg_thresh = torch.min(reg_fg_thresh, cls_fg_thresh)
        else:
            fg_thresh = min(reg_fg_thresh, cls_fg_thresh)

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)  # > 0.55
        easy_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)  # < 0.1
        hard_bg_inds = ((max_overlaps < fg_thresh) &
                (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = fg_inds[fg_inds < 0] # yield empty tensor

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    """
    Local IoU-based filtering for unlabeled samples.
    """
    def classwise_iou_filtering(self, gt_of_rois, gt_iou_of_rois, roi_labels, rcnn_cls_score_teacher=None):
        # Note that the weights are defined only for unlabeled samples. All labeled samples receive one as weight.

        gt_iou_of_rois = gt_iou_of_rois.detach().clone()
        roi_labels = roi_labels.detach().clone() - 1

        # ----------- REG_VALID_MASK -----------
        ulb_reg_fg_thresh = self.roi_sampler_cfg.UNLABELED_REG_FG_THRESH
        ulb_reg_fg_thresh = gt_iou_of_rois.new_tensor(ulb_reg_fg_thresh).reshape(1, 1, -1).repeat(
            *gt_iou_of_rois.shape[:2], 1)
        ulb_reg_fg_thresh = torch.gather(ulb_reg_fg_thresh, dim=-1, index=roi_labels.unsqueeze(-1)).squeeze(-1)
        if self.roi_sampler_cfg.get("UNLABELED_TEACHER_SCORES_FOR_RVM", False):
            # TODO Ensure this works with new classwise thresholds
            # filtering_mask = ret_dict['rcnn_cls_score_teacher'] > ulb_reg_fg_thresh
            filtering_mask = rcnn_cls_score_teacher > ulb_reg_fg_thresh
        else:
            filtering_mask = gt_iou_of_rois > ulb_reg_fg_thresh
        reg_valid_mask = filtering_mask.long()

        # ----------- RCNN_CLS_LABELS -----------
        ulb_cls_fg_thresh = self.roi_sampler_cfg.UNLABELED_CLS_FG_THRESH
        fg_thresh = gt_iou_of_rois.new_tensor(ulb_cls_fg_thresh).reshape(1, 1, -1).repeat(
            *gt_iou_of_rois.shape[:2], 1)
        cls_fg_thresh = torch.gather(fg_thresh, dim=-1, index=roi_labels.unsqueeze(-1)).squeeze(-1)
        cls_bg_thresh = self.roi_sampler_cfg.UNLABELED_CLS_BG_THRESH

        ulb_fg_mask = gt_iou_of_rois > cls_fg_thresh
        ulb_bg_mask = gt_iou_of_rois < cls_bg_thresh
        interval_mask = ~(ulb_fg_mask | ulb_bg_mask)
        ignore_mask = torch.eq(gt_of_rois, 0).all(dim=-1)

        # Hard labeling for FGs/BGs, soft labeling for UCs
        gt_iou_of_rois[ignore_mask] = -1
        if self.roi_sampler_cfg.get("UNLABELED_SHARP_RCNN_CLS_LABELS", False):
            gt_iou_of_rois[ulb_fg_mask] = 1.
            gt_iou_of_rois[ulb_bg_mask] = 0.
        # Calibrate(normalize) raw IoUs as per FG and BG thresholds
        if self.roi_sampler_cfg.get("UNLABELED_USE_CALIBRATED_IOUS", False):
            gt_iou_of_rois[interval_mask] = (gt_iou_of_rois[interval_mask] - cls_bg_thresh) \
                                                / (cls_fg_thresh[interval_mask] - cls_bg_thresh)
        rcnn_cls_labels = gt_iou_of_rois

        return rcnn_cls_labels, reg_valid_mask, interval_mask

    def get_reliability_weights(self, rcnn_cls_labels, rcnn_cls_score_teacher, interval_mask):
        rcnn_cls_weights = torch.ones_like(rcnn_cls_labels)
        weight_type = self.roi_sampler_cfg['UNLABELED_RELIABILITY_WEIGHT_TYPE']
        rcnn_bg_score_teacher = 1 - rcnn_cls_score_teacher  # represents the bg score
        if weight_type == 'all':
            rcnn_cls_weights[interval_mask] = rcnn_bg_score_teacher[interval_mask]
        elif weight_type == 'interval-only':
            unlabeled_rcnn_cls_weights = torch.zeros_like(rcnn_cls_labels)
            unlabeled_rcnn_cls_weights[interval_mask] = rcnn_bg_score_teacher[interval_mask]
        elif weight_type == 'bg':
            ulb_bg_mask = rcnn_cls_labels == 0
            rcnn_cls_weights[ulb_bg_mask] = rcnn_bg_score_teacher[ulb_bg_mask]
        elif weight_type == 'ignore_interval':  # Naive baseline
            rcnn_cls_weights[interval_mask] = 0
        elif weight_type == 'full-ema':
            unlabeled_rcnn_cls_weights = rcnn_bg_score_teacher
        # Use 1s for FG mask, teacher's BG scores for UC+BG mask
        elif weight_type == 'uc-bg':
            ulb_fg_mask = rcnn_cls_labels == 1
            rcnn_cls_weights[~ulb_fg_mask] = rcnn_bg_score_teacher[~ulb_fg_mask]
        # Use teacher's FG scores for FG mask, teacher's BG scores for UC+BG mask
        elif weight_type == 'fg-uc-bg':
            ulb_fg_mask = rcnn_cls_labels == 1
            rcnn_cls_weights[ulb_fg_mask] = rcnn_cls_score_teacher[ulb_fg_mask]
            rcnn_cls_weights[~ulb_fg_mask] = rcnn_bg_score_teacher[~ulb_fg_mask]
        else:
            raise ValueError

        return rcnn_cls_weights