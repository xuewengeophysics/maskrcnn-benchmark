# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor  ##self.box为build_roi_box_head中的ROIBoxHead(cfg, in_channels)
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    ##features为generalized_rcnn.py中的self.backbone(images.tensors)
    ##proposals为generalized_rcnn.py中的self.rpn(images, features, targets)
    ##targets为maskrcnn_benchmark/data/datasets/coco.py中的class COCODataset中的__getitem__(self, idx)
    def forward(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        ##self.box为build_roi_box_head中的ROIBoxHead(cfg, in_channels)
        ##ROIBoxHead返回x, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            ##self.mask为build_roi_mask_head中的ROIMaskHead(cfg, in_channels)
            ##ROIMaskHead返回x, all_proposals, dict(loss_mask=loss_mask)
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            ##self.keypoint为build_roi_keypoint_head中的ROIKeypointHead(cfg, in_channels)
            ##ROIKeypointHead返回x, proposals, dict(loss_kp=loss_kp)
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)
        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    ##_C.MODEL.RPN_ONLY = False(maskrcnn_benchmark/config/defaults.py)
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
