import torch
from .pv_rcnn import PVRCNN
from pcdet.utils.stats_utils import MetricRegistry

class ReliableStudentPVRCNN(PVRCNN):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.model_type = None
        self.accumulated_itr = 0
        self.metric_registry = MetricRegistry(dataset=self.dataset, model_cfg=model_cfg)
        vals_to_store = ['iou_roi_pl', 'iou_roi_gt', 'pred_scores', 'weights', 'class_labels', 'iteration']
        self.val_dict = {val: [] for val in vals_to_store}

    def set_model_type(self, model_type):
        assert model_type in ['origin', 'teacher', 'student']
        self.model_type = model_type
        self.dense_head.model_type = model_type
        self.roi_head.model_type = model_type

    def forward(self, batch_dict):

        if self.model_type == 'origin':
            return super().forward(batch_dict)

        elif self.model_type == 'teacher':
            with torch.no_grad():
                for cur_module in self.module_list:
                    batch_dict = cur_module(batch_dict)
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

        elif self.model_type == 'student':
            if self.training:
                if batch_dict['type'] == 'labeled':
                    return super().forward(batch_dict)
                else:
                    raise ValueError
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts

    def get_training_loss_ulb(self):

        tb_dict, disp_dict = {}, {}

        # RPN Loss
        rpn_loss = 0
        rpn_box_loss, rpn_box_tb_dict = self.dense_head.get_box_reg_layer_loss()
        tb_dict.update(rpn_box_tb_dict)
        rpn_loss += rpn_box_loss
        if self.model_cfg.ROI_HEAD.get('UNLABELED_SUPERVISE_CLS', False):
            rpn_cls_loss, rpn_cls_tb_dict = self.dense_head.get_cls_layer_loss()
            rpn_loss += rpn_cls_loss
            tb_dict.update(rpn_cls_tb_dict)
        tb_dict['rpn_loss'] = rpn_loss

        rcnn_loss, rcnn_tb_dict = self.roi_head.get_loss_ulb(tb_dict)

        # Point Loss
        point_loss, point_tb_dict = self.point_head.get_loss(tb_dict)

        loss = point_loss + rpn_loss + rcnn_loss
        tb_dict['loss'] = loss

        ub_tb_dict = {}
        for key in tb_dict:
            if 'loss' in key or 'point_pos_num' in key:
                ub_tb_dict[f"{key}_unlabeled"] = tb_dict[key]
            else:
                ub_tb_dict[key] = tb_dict[key]

        if self.model_cfg.ROI_HEAD.get("ENABLE_EVAL", None):
            metrics_pred_types = self.model_cfg.ROI_HEAD.get("METRICS_PRED_TYPES", None)
            self.update_metrics(self.roi_head.forward_ret_dict, pred_type=metrics_pred_types)

        for key in self.metric_registry.tags():
            metrics = self.compute_metrics(tag=key)
            tb_dict.update(metrics)

        ret_dict = {'loss': loss}
        return ret_dict, ub_tb_dict, disp_dict

    def update_metrics(self, targets_dict, mask_type='cls', vis_type='pred_gt', pred_type=None):
        sample_preds, sample_pred_scores, sample_pred_weights = [], [], []
        sample_rois, sample_roi_scores = [], []
        sample_targets, sample_target_scores = [], []
        sample_pls, sample_pl_scores = [], []
        ema_preds_of_std_rois, ema_pred_scores_of_std_rois = [], []
        # sample_gts = []
        sample_gt_iou_of_rois = []
        for ind in range(targets_dict['rois'].shape[0]):
            mask = (targets_dict['reg_valid_mask'][ind] > 0) if mask_type == 'reg' else (
                    targets_dict['rcnn_cls_labels'][ind] >= 0)

            # (Proposals) ROI info
            rois = targets_dict['rois'][ind][mask].detach().clone()
            roi_labels = targets_dict['roi_labels'][ind][mask].unsqueeze(-1).detach().clone()
            roi_scores = torch.sigmoid(targets_dict['roi_scores'])[ind][mask].detach().clone()
            roi_labeled_boxes = torch.cat([rois, roi_labels], dim=-1)
            gt_iou_of_rois = targets_dict['gt_iou_of_rois'][ind][mask].unsqueeze(-1).detach().clone()
            sample_rois.append(roi_labeled_boxes)
            sample_roi_scores.append(roi_scores)
            sample_gt_iou_of_rois.append(gt_iou_of_rois)
            # Target info
            target_labeled_boxes = targets_dict['gt_of_rois_src'][ind][mask].detach().clone()
            target_scores = targets_dict['rcnn_cls_labels'][ind][mask].detach().clone()
            sample_targets.append(target_labeled_boxes)
            sample_target_scores.append(target_scores)

            # Pred info
            pred_boxes = targets_dict['batch_box_preds'][ind][mask].detach().clone()
            pred_scores = torch.sigmoid(targets_dict['rcnn_cls']).view_as(targets_dict['rcnn_cls_labels'])[ind][
                mask].detach().clone()
            pred_labeled_boxes = torch.cat([pred_boxes, roi_labels], dim=-1)
            sample_preds.append(pred_labeled_boxes)
            sample_pred_scores.append(pred_scores)

            # (Real labels) GT info
            # gt_labeled_boxes = targets_dict['org_gt_boxes'][ind]
            # sample_gts.append(gt_labeled_boxes)

            # (Pseudo labels) PL info
            pl_labeled_boxes = targets_dict['gt_boxes'][ind]
            pl_scores = targets_dict['pl_scores'][ind]
            sample_pls.append(pl_labeled_boxes)
            sample_pl_scores.append(pl_scores)

            # Teacher refinements (Preds) of student's rois
            if 'ema_gt' in pred_type and self.model_cfg.ROI_HEAD.get('ENABLE_SOFT_TEACHER', False):
                # pred_boxes_ema = targets_dict['batch_box_preds_teacher'][ind][mask].detach().clone()
                # pred_labeled_boxes_ema = torch.cat([pred_boxes_ema, roi_labels], dim=-1)
                # pred_scores_ema = targets_dict['rcnn_cls_score_teacher'][ind][mask].detach().clone()
                # ema_preds_of_std_rois.append(pred_labeled_boxes_ema)
                # ema_pred_scores_of_std_rois.append(pred_scores_ema)
                raise NotImplementedError
            if self.model_cfg.ROI_HEAD.get('ENABLE_SOFT_TEACHER', False):
                rcnn_cls_weights = targets_dict['rcnn_cls_weights'].view_as(targets_dict['rcnn_cls_labels'])
                pred_weights = rcnn_cls_weights[ind][mask].detach().clone()
                sample_pred_weights.append(pred_weights)
            if self.model_cfg.ROI_HEAD.get('ENABLE_VIS', False):
                # points_mask = targets_dict['points'][:, 0] == ind
                # points = targets_dict['points'][points_mask, 1:]
                # gt_boxes = gt_labeled_boxes[:, :-1]  # Default GT option

                if vis_type == 'roi_gt':
                    vis_pred_boxes = roi_labeled_boxes[:, :-1]
                    vis_pred_scores = roi_scores
                elif vis_type == 'roi_pl':
                    vis_pred_boxes = roi_labeled_boxes[:, :-1]
                    vis_pred_scores = roi_scores
                    gt_boxes = pl_labeled_boxes[:, :-1]
                elif vis_type == 'target_gt':
                    vis_pred_boxes = target_labeled_boxes[:, :-1]
                    vis_pred_scores = target_scores
                elif vis_type == 'pred_gt':
                    vis_pred_boxes = pred_boxes
                    vis_pred_scores = pred_scores
                elif vis_type == 'ema_gt' and self.model_cfg.ROI_HEAD.get('ENABLE_SOFT_TEACHER', False):
                    vis_pred_boxes = targets_dict['batch_box_preds_teacher'][ind][mask].detach().clone()
                    vis_pred_scores = targets_dict['rcnn_cls_score_teacher'][ind][mask].detach().clone()
                else:
                    raise ValueError(vis_type)

                # V.vis(points, gt_boxes=gt_boxes, pred_boxes=vis_pred_boxes,
                #       pred_scores=vis_pred_scores, pred_labels=roi_labels.view(-1),
                #       filename=f'vis_{vis_type}_{uind}.png')

        sample_pred_weights = sample_pred_weights if len(sample_pred_weights) > 0 else None

        if 'ema_gt' in pred_type and self.model_cfg.ROI_HEAD.get('ENABLE_SOFT_TEACHER', False):
            # tag = f'rcnn_ema_gt_metrics_{mask_type}'
            # metrics_ema = self.metric_registry.get(tag)
            # metric_inputs_ema = {'preds': ema_preds_of_std_rois, 'pred_scores': ema_pred_scores_of_std_rois,
            #                      'ground_truths': sample_gts, 'pred_weights': sample_pred_weights}
            # metrics_ema.update(**metric_inputs_ema)
            raise NotImplementedError
        if 'roi_pl' in pred_type:
            tag = f'rcnn_roi_pl_metrics_{mask_type}'
            metrics_roi_pl = self.metric_registry.get(tag)
            metric_inputs_roi_pl = {'preds': sample_rois, 'pred_scores': sample_roi_scores, 'ground_truths': sample_pls,
                                    'targets': sample_targets, 'target_scores': sample_target_scores,
                                    'pred_weights': sample_pred_weights}
            metrics_roi_pl.update(**metric_inputs_roi_pl)
        if 'pred_pl' in pred_type:
            tag = f'rcnn_pred_pl_metrics_{mask_type}'
            metrics_pred_pl = self.metric_registry.get(tag)
            metric_inputs_pred_pl = {'preds': sample_preds, 'pred_scores': sample_pred_scores, 'rois': sample_rois,
                                     'roi_scores': sample_roi_scores, 'ground_truths': sample_pls,
                                     'targets': sample_targets, 'target_scores': sample_target_scores,
                                     'pred_weights': sample_pred_weights}
            metrics_pred_pl.update(**metric_inputs_pred_pl)
        if 'roi_pl_gt' in pred_type:
            tag = f'rcnn_roi_pl_gt_metrics_{mask_type}'
            # metrics = self.metric_registry.get(tag)
            # metric_inputs = {'preds': sample_rois, 'pred_scores': sample_roi_scores,
            #                  'ground_truths': sample_gts, 'targets': sample_targets,
            #                  'pseudo_labels': sample_pls, 'pseudo_label_scores': sample_pl_scores,
            #                  'target_scores': sample_target_scores, 'pred_weights': sample_pred_weights,
            #                  'pred_iou_wrt_pl': sample_gt_iou_of_rois}
            # metrics.update(**metric_inputs)
            raise NotImplementedError

    @torch.no_grad()
    def compute_metrics(self, tag):
        results = self.metric_registry.get(tag).compute()
        tag = f"{tag}/" if tag else ''
        return {tag + key: val for key, val in results.items()}