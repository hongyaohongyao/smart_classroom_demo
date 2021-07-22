import numpy as np
from cv2 import cv2


class CropImage:
    @staticmethod
    def get_new_box(src_w, src_h, bbox, scale, out_w, out_h):
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]
        center_x, center_y = box_w / 2 + x, box_h / 2 + y
        aspect_src = box_w / box_h
        aspect_target = out_w / out_h
        # 调整边框比例
        if aspect_src > aspect_target:
            box_h = box_w / aspect_target
        else:
            box_w = box_h * aspect_target

        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale

        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2
        # 调整边框位置
        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1

        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        return int(left_top_x), int(left_top_y), \
               int(right_bottom_x), int(right_bottom_y)

    @staticmethod
    def crop(org_img, bbox, scale, out_w, out_h, crop=True, return_box=False):

        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h))
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = bbox
        else:
            src_h, src_w, _ = np.shape(org_img)
            left_top_x, left_top_y, \
            right_bottom_x, right_bottom_y = CropImage.get_new_box(src_w, src_h, bbox, scale, out_w, out_h)

            img = org_img[left_top_y: right_bottom_y + 1,
                  left_top_x: right_bottom_x + 1]
            dst_img = cv2.resize(img, (out_w, out_h))
        return (dst_img, [left_top_x, left_top_y, right_bottom_x, right_bottom_y]) if return_box else dst_img
