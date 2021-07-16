# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com), Haoyi Zhu
# -----------------------------------------------------

import cv2
import numpy as np
import torch

try:
    from alphapose.bbox import (_box_to_center_scale, _center_scale_to_box)
    from alphapose.transforms import (get_affine_transform, im_to_torch)
except ImportError:
    from utils.alphapose.bbox import (_box_to_center_scale, _center_scale_to_box)
    from utils.alphapose.transforms import (get_affine_transform, im_to_torch)


# Only windows visual studio 2013 ~2017 support compile c/cuda extensions
# If you force to compile extension on Windows and ensure appropriate visual studio
# is intalled, you can try to use these ext_modules.
# if platform.system() != 'Windows':
#     from ..roi_align import RoIAlign


class SimpleTransform(object):
    """Generation of cropped input person and pose heatmaps from SimplePose.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, h, w)`.
    label: dict
        A dictionary with 4 keys:
            `bbox`: [xmin, ymin, xmax, ymax]
            `joints_3d`: numpy.ndarray with shape: (n_joints, 2),
                    including position and visible flag
            `width`: image width
            `height`: image height
    dataset:
        The dataset to be transformed, must include `joint_pairs` property for flipping.
    scale_factor: int
        Scale augmentation.
    input_size: tuple
        Input image size, as (height, width).
    output_size: tuple
        Heatmap size, as (height, width).
    rot: int
        Ratation augmentation.
    train: bool
        True for training trasformation.
    """

    def __init__(self, scale_factor, add_dpg,
                 input_size, output_size, rot, sigma,
                 gpu_device=None, loss_type='MSELoss'):
        self._scale_factor = scale_factor
        self._rot = rot
        self._add_dpg = add_dpg
        self._gpu_device = gpu_device

        self._input_size = input_size
        self._heatmap_size = output_size

        self._sigma = sigma
        self._loss_type = loss_type
        self._aspect_ratio = float(input_size[1]) / input_size[0]  # w / h
        self._feat_stride = np.array(input_size) / np.array(output_size)

        self.pixel_std = 1

    def test_transform(self, src, bbox):
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio)
        scale = scale * 1.0

        input_size = self._input_size
        inp_h, inp_w = input_size

        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)

        img = im_to_torch(img)
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        return img, bbox

    def _target_generator(self, joints_3d, num_joints):
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target = np.zeros((num_joints, self._heatmap_size[0], self._heatmap_size[1]),
                          dtype=np.float32)
        tmp_size = self._sigma * 3

        for i in range(num_joints):
            mu_x = int(joints_3d[i, 0, 0] / self._feat_stride[0] + 0.5)
            mu_y = int(joints_3d[i, 1, 0] / self._feat_stride[1] + 0.5)
            # check if any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self._heatmap_size[1] or ul[1] >= self._heatmap_size[0] or br[0] < 0 or br[1] < 0:
                # return image as is
                target_weight[i] = 0
                continue

            # generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # the gaussian is not normalized, we want the center value to be equal to 1
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self._sigma ** 2)))

            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self._heatmap_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self._heatmap_size[0]) - ul[1]
            # image range
            img_x = max(0, ul[0]), min(br[0], self._heatmap_size[1])
            img_y = max(0, ul[1]), min(br[1], self._heatmap_size[0])

            v = target_weight[i]
            if v > 0.5:
                target[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, np.expand_dims(target_weight, -1)

    @staticmethod
    def _integral_target_generator(joints_3d, num_joints, patch_height, patch_width, source=None):
        target_weight = np.ones((num_joints, 2), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]
        if num_joints == 136:
            target_weight[:26, :] = target_weight[:26, :] * 2
        elif num_joints == 133:
            target_weight[:23, :] = target_weight[:23, :] * 2

        if source == 'frei' or source == 'partX' or source == 'OneHand' or source == 'hand_labels_synth' \
                or source == 'hand143_panopticdb' or source == 'RHD_published_v2' or source == 'interhand':
            if target_weight[-21:, :].sum() > 0 and target_weight[-42:-21].sum() == 0:
                target_weight[-42:-21] += 1
            elif target_weight[-21:, :].sum() == 0 and target_weight[-42:-21].sum() > 0:
                target_weight[-21:, :] += 1

        target = np.zeros((num_joints, 2), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight
