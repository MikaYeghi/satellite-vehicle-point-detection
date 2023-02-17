import torch
from detectron2.layers import cat
from torch.nn import functional as F
from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils.events import get_event_storage
from detectron2.modeling.meta_arch.retinanet import RetinaNet

from .losses import _dense_box_regression_loss

@META_ARCH_REGISTRY.register()
class RetinaNetPoint(RetinaNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.
        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor storing the loss.
                Used during training only. The dict keys are: "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        normalizer = self._ema_update("loss_normalizer", max(num_pos_anchors, 1), 100)

        # classification and regression loss
        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[
            :, :-1
        ]  # no loss for the last (background) class
        loss_cls = sigmoid_focal_loss_jit(
            cat(pred_logits, dim=1)[valid_mask],
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        loss_point_reg = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        # # Normalize the loss terms
        # class_loss = loss_cls / normalizer
        # if self.box_reg_loss_type == "l2_point":
        #     point_regression_loss = 100 * loss_point_reg
        # elif self.box_reg_loss_type == "smooth_l1_point":
        #     class_loss *= 0.1
        #     point_regression_loss = loss_point_reg / normalizer
        # elif self.box_reg_loss_type == "t_distr_point":
        #     point_regression_loss = loss_point_reg
        # else:
        #     raise NotImplementedError
        
        return {
            "loss_cls": loss_cls / normalizer,
            "loss_box_reg": loss_point_reg / normalizer,
        }