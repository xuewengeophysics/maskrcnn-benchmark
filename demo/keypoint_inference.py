# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

import cv2
# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')

from predictor import COCODemo

config_file = "./configs/hrnet/e2e_keypoint_rcnn_hrnet_w18_1x.yaml"
# config_file = "./configs/e2e_keypoint_rcnn_R_50_FPN_1x_train2017.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image_filename = "/home/wenxue/SRC/projects/keypoint_detection/coco/images/val2017/000000053626.jpg"
image = cv2.imread(image_filename)
predictions = coco_demo.run_on_opencv_image(image)
cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, 'kps_' + image_filename.split("/")[-1]), predictions)
import ipdb;ipdb.set_trace()