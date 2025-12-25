# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from ..utils import LOGGER, colorstr
from ..utils.checks import check_version
from ..utils.instance import Instances
from ..utils.metrics import bbox_ioa
from ..utils.ops import segment2box
from .utils import polygons2masks, polygons2masks_overlap

POSE_FLIPLR_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


# TODO: we might need a BaseTransform to make all these augments be compatible with both classification and semantic
class BaseTransform:

    def __init__(self) -> None:
        pass

    def apply_image(self, labels):
        """Applies image transformation to labels."""
        pass

    def apply_instances(self, labels):
        """Applies transformations to input 'labels' and returns object instances."""
        pass

    def apply_semantic(self, labels):
        """Applies semantic segmentation to an image."""
        pass

    def __call__(self, labels):
        """Applies label transformations to an image, instances and semantic masks."""
        self.apply_image(labels)
        self.apply_instances(labels)
        self.apply_semantic(labels)


class Compose:

    def __init__(self, transforms):
        """Initializes the Compose object with a list of transforms."""
        self.transforms = transforms

    def __call__(self, data):
        """Applies a series of transformations to input data."""
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """Appends a new transform to the existing list of transforms."""
        self.transforms.append(transform)

    def tolist(self):
        """Converts list of transforms to a standard Python list."""
        return self.transforms

    def __repr__(self):
        """Return string representation of object."""
        format_string = f'{self.__class__.__name__}('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class BaseMixTransform:
    """This implementation is from mmyolo."""

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels):
        """Applies pre-processing transforms and mixup/mosaic transforms to labels data."""
        if random.uniform(0, 1) > self.p:
            return labels

        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for Mosaic or MixUp
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels['mix_labels'] = mix_labels

        # Mosaic or MixUp
        labels = self._mix_transform(labels)
        labels.pop('mix_labels', None)
        return labels

    def _mix_transform(self, labels):
        """Applies MixUp or Mosaic augmentation to the label dictionary."""
        raise NotImplementedError

    def get_indexes(self):
        """Gets a list of shuffled indexes for mosaic augmentation."""
        raise NotImplementedError


class Mosaic(BaseMixTransform):
    """
    Mosaic augmentation.

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
    The augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int, optional): Image size (height and width) after mosaic pipeline of a single image. Default to 640.
        p (float, optional): Probability of applying the mosaic augmentation. Must be in the range 0-1. Default to 1.0.
        n (int, optional): The grid size, either 4 (for 2x2) or 9 (for 3x3).
    """

    def __init__(self, dataset, imgsz=640, p=1.0, n=4, cube=False):
        """Initializes the object with a dataset, image size, probability, and border."""
        assert 0 <= p <= 1.0, f'The probability should be in range [0, 1], but got {p}.'
        assert n in (4, 9), 'grid must be equal to 4 or 9.'
        super().__init__(dataset=dataset, p=p)
        self.dataset = dataset
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = n
        self.cube = cube

    def get_indexes(self, buffer=True):
        """Return a list of random indexes from the dataset."""
        if buffer:  # select images from buffer
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self, labels):
        """Apply mixup transformation to the input image and labels."""
        assert labels.get('rect_shape', None) is None, 'rect and mosaic are mutually exclusive.'
        assert len(labels.get('mix_labels', [])), 'There are no other images for mosaic augment.'
        return self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)

    def _mosaic4(self, labels):
        """Create a 2x2 image mosaic."""
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
        for i in range(4):
            labels_patch = labels if i == 0 else labels['mix_labels'][i - 1]
            # Load image
            img = labels_patch['img']
            h, w = labels_patch.pop('resized_shape')

            # Place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh, self.cube)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels['img'] = img4
        return final_labels

    def _mosaic9(self, labels):
        """Create a 3x3 image mosaic."""
        mosaic_labels = []
        s = self.imgsz
        hp, wp = -1, -1  # height, width previous
        for i in range(9):
            labels_patch = labels if i == 0 else labels['mix_labels'][i - 1]
            # Load image
            img = labels_patch['img']
            h, w = labels_patch.pop('resized_shape')

            # Place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Image
            img9[y1:y2, x1:x2] = img[y1 - padh:, x1 - padw:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming imgsz*2 mosaic size
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1], self.cube)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels['img'] = img9[-self.border[0]:self.border[0], -self.border[1]:self.border[1]]
        return final_labels

    @staticmethod
    def _update_labels(labels, padw, padh, cube):
        """Update labels."""
        nh, nw = labels['img'].shape[:2]
        if cube:
            labels['instances_t1'].convert_bbox(format='xyxy')
            labels['instances_t1'].denormalize(nw, nh)
            labels['instances_t1'].add_padding(padw, padh)

            labels['instances_t2'].convert_bbox(format='xyxy')
            labels['instances_t2'].denormalize(nw, nh)
            labels['instances_t2'].add_padding(padw, padh)

            labels['instances_t3'].convert_bbox(format='xyxy')
            labels['instances_t3'].denormalize(nw, nh)
            labels['instances_t3'].add_padding(padw, padh)

            # labels['instances_t4'].convert_bbox(format='xyxy')
            # labels['instances_t4'].denormalize(nw, nh)
            # labels['instances_t4'].add_padding(padw, padh)

            # labels['instances_t5'].convert_bbox(format='xyxy')
            # labels['instances_t5'].denormalize(nw, nh)
            # labels['instances_t5'].add_padding(padw, padh)

            # labels['instances_t6'].convert_bbox(format='xyxy')
            # labels['instances_t6'].denormalize(nw, nh)
            # labels['instances_t6'].add_padding(padw, padh)
        else:
            labels['instances'].convert_bbox(format='xyxy')
            labels['instances'].denormalize(nw, nh)
            labels['instances'].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels):
        """Return labels with mosaic border instances clipped."""
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        if self.cube:
            instances_t1 = []
            instances_t2 = []
            instances_t3 = []
            # instances_t4 = []
            # instances_t5 = []
            # instances_t6 = []
            imgsz = self.imgsz * 2  # mosaic imgsz
            for labels in mosaic_labels:
                cls.append(labels['cls'])
                instances_t1.append(labels['instances_t1'])
                instances_t2.append(labels['instances_t2'])
                instances_t3.append(labels['instances_t3'])
                # instances_t4.append(labels['instances_t4'])
                # instances_t5.append(labels['instances_t5'])
                # instances_t6.append(labels['instances_t6'])
            final_labels = {
                'im_file': mosaic_labels[0]['im_file'],
                'ori_shape': mosaic_labels[0]['ori_shape'],
                'resized_shape': (imgsz, imgsz),
                'cls': np.concatenate(cls, 0),
                'instances_t1': Instances.concatenate(instances_t1, axis=0),
                'instances_t2': Instances.concatenate(instances_t2, axis=0),
                'instances_t3': Instances.concatenate(instances_t3, axis=0),
                # 'instances_t4': Instances.concatenate(instances_t4, axis=0),
                # 'instances_t5': Instances.concatenate(instances_t5, axis=0),
                # 'instances_t6': Instances.concatenate(instances_t6, axis=0),
                'mosaic_border': self.border}  # final_labels
            final_labels['instances_t1'].clip(imgsz, imgsz)
            final_labels['instances_t2'].clip(imgsz, imgsz)
            final_labels['instances_t3'].clip(imgsz, imgsz)
            # final_labels['instances_t4'].clip(imgsz, imgsz)
            # final_labels['instances_t5'].clip(imgsz, imgsz)
            # final_labels['instances_t6'].clip(imgsz, imgsz)
            good = final_labels['instances_t3'].remove_zero_area_boxes()
            final_labels['instances_t1'] = final_labels['instances_t1'][good]
            final_labels['instances_t2'] = final_labels['instances_t2'][good]
            # final_labels['instances_t3'] = final_labels['instances_t3'][good]
            # final_labels['instances_t4'] = final_labels['instances_t4'][good]
            # final_labels['instances_t5'] = final_labels['instances_t5'][good]
            final_labels['cls'] = final_labels['cls'][good]
        else:
            instances = []
            for labels in mosaic_labels:
                cls.append(labels['cls'])
                instances.append(labels['instances'])
            final_labels = {
                'im_file': mosaic_labels[0]['im_file'],
                'ori_shape': mosaic_labels[0]['ori_shape'],
                'resized_shape': (imgsz, imgsz),
                'cls': np.concatenate(cls, 0),
                'instances': Instances.concatenate(instances, axis=0),
                'mosaic_border': self.border}  # final_labels
            final_labels['instances'].clip(imgsz, imgsz)
            good = final_labels['instances'].remove_zero_area_boxes()
            final_labels['cls'] = final_labels['cls'][good]
        return final_labels


class MixUp(BaseMixTransform):

    def __init__(self, dataset, pre_transform=None, p=0.0, cube=False) -> None:
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        self.cube = cube

    def get_indexes(self):
        """Get a random index from the dataset."""
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        """Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf."""
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        labels2 = labels['mix_labels'][0]
        labels['img'] = (labels['img'] * r + labels2['img'] * (1 - r)).astype(np.uint8)
        if self.cube:
            labels['instances_t1'] = Instances.concatenate([labels['instances_t1'], labels2['instances_t1']], axis=0)
            labels['instances_t2'] = Instances.concatenate([labels['instances_t2'], labels2['instances_t2']], axis=0)
            labels['instances_t3'] = Instances.concatenate([labels['instances_t3'], labels2['instances_t3']], axis=0)
            # labels['instances_t4'] = Instances.concatenate([labels['instances_t4'], labels2['instances_t4']], axis=0)
            # labels['instances_t5'] = Instances.concatenate([labels['instances_t5'], labels2['instances_t5']], axis=0)
            # labels['instances_t6'] = Instances.concatenate([labels['instances_t6'], labels2['instances_t6']], axis=0)
        else:
            labels['instances'] = Instances.concatenate([labels['instances'], labels2['instances']], axis=0)
        labels['cls'] = np.concatenate([labels['cls'], labels2['cls']], 0)
        return labels


class RandomPerspective:

    def __init__(self,
                 degrees=0.0,
                 translate=0.1,
                 scale=0.5,
                 shear=0.0,
                 perspective=0.0,
                 border=(0, 0),
                 pre_transform=None,
                 cube=False):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        # Mosaic border
        self.border = border
        self.pre_transform = pre_transform
        self.cube = cube

    def affine_transform(self, img, border):
        """Center."""
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if img.shape[2] <= 4:
                border_val = (114,) * img.shape[2]
                if self.perspective:
                    img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=border_val)
                else:  # affine
                    img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=border_val)
            else:
                # Split and warp separately for > 4 channels (OpenCV limit)
                img_parts = []
                for i in range(0, img.shape[2], 4):
                    chunk = img[:, :, i:i+4]
                    chunk_border_val = (114,) * chunk.shape[2]
                    if self.perspective:
                        p = cv2.warpPerspective(chunk, M, dsize=self.size, borderValue=chunk_border_val)
                    else:
                        p = cv2.warpAffine(chunk, M[:2], dsize=self.size, borderValue=chunk_border_val)
                    img_parts.append(p)
                img = np.concatenate(img_parts, axis=2)
        return img, M, s

    def apply_bboxes(self, bboxes, M):
        """
        Apply affine to bboxes only.

        Args:
            bboxes (ndarray): list of bboxes, xyxy format, with shape (num_bboxes, 4).
            M (ndarray): affine matrix.

        Returns:
            new_bboxes (ndarray): bboxes after affine, [num_bboxes, 4].
        """
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def apply_segments(self, segments, M):
        """
        Apply affine to segments and generate new bboxes from segments.

        Args:
            segments (ndarray): list of segments, [num_samples, 500, 2].
            M (ndarray): affine matrix.

        Returns:
            new_segments (ndarray): list of segments after affine, [num_samples, 500, 2].
            new_bboxes (ndarray): bboxes after affine, [N, 4].
        """
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments

        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack([segment2box(xy, self.size[0], self.size[1]) for xy in segments], 0)
        return bboxes, segments

    def apply_keypoints(self, keypoints, M):
        """
        Apply affine to keypoints.

        Args:
            keypoints (ndarray): keypoints, [N, 17, 3].
            M (ndarray): affine matrix.

        Return:
            new_keypoints (ndarray): keypoints after affine, [N, 17, 3].
        """
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]  # perspective rescale or affine
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > self.size[0]) | (xy[:, 1] > self.size[1])
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def __call__(self, labels):
        """
        Affine images and targets.

        Args:
            labels (dict): a dict of `bboxes`, `segments`, `keypoints`.
        """
        if self.pre_transform and 'mosaic_border' not in labels:
            labels = self.pre_transform(labels)
        labels.pop('ratio_pad', None)  # do not need ratio pad

        img = labels['img']
        cls = labels['cls']
        if self.cube:
            instances_t1 = labels.pop('instances_t1')
            instances_t1.convert_bbox(format='xyxy')
            instances_t1.denormalize(*img.shape[:2][::-1])

            instances_t2 = labels.pop('instances_t2')
            instances_t2.convert_bbox(format='xyxy')
            instances_t2.denormalize(*img.shape[:2][::-1])

            instances_t3 = labels.pop('instances_t3')
            instances_t3.convert_bbox(format='xyxy')
            instances_t3.denormalize(*img.shape[:2][::-1])

            # instances_t4 = labels.pop('instances_t4')
            # instances_t4.convert_bbox(format='xyxy')
            # instances_t4.denormalize(*img.shape[:2][::-1])

            # instances_t5 = labels.pop('instances_t5')
            # instances_t5.convert_bbox(format='xyxy')
            # instances_t5.denormalize(*img.shape[:2][::-1])

            # instances_t6 = labels.pop('instances_t6')
            # instances_t6.convert_bbox(format='xyxy')
            # instances_t6.denormalize(*img.shape[:2][::-1])
        else:
            instances = labels.pop('instances')
            # Make sure the coord formats are right
            instances.convert_bbox(format='xyxy')
            instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop('mosaic_border', self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M is affine matrix
        # scale for func:`box_candidates`
        img, M, scale = self.affine_transform(img, border)

        if self.cube:
            bboxes_t1 = self.apply_bboxes(instances_t1.bboxes, M)
            bboxes_t2 = self.apply_bboxes(instances_t2.bboxes, M)
            bboxes_t3 = self.apply_bboxes(instances_t3.bboxes, M)
            # bboxes_t4 = self.apply_bboxes(instances_t4.bboxes, M)
            # bboxes_t5 = self.apply_bboxes(instances_t5.bboxes, M)
            # bboxes_t6 = self.apply_bboxes(instances_t6.bboxes, M)
            segments = instances_t3.segments
            keypoints = instances_t3.keypoints
        else:
            bboxes = self.apply_bboxes(instances.bboxes, M)
            segments = instances.segments
            keypoints = instances.keypoints
        # Update bboxes if there are segments.
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        
        if self.cube:
            new_instances_t1 = Instances(bboxes_t1, segments, keypoints, bbox_format='xyxy', normalized=False)
            new_instances_t2 = Instances(bboxes_t2, segments, keypoints, bbox_format='xyxy', normalized=False)
            new_instances_t3 = Instances(bboxes_t3, segments, keypoints, bbox_format='xyxy', normalized=False)
            # new_instances_t4 = Instances(bboxes_t4, segments, keypoints, bbox_format='xyxy', normalized=False)
            # new_instances_t5 = Instances(bboxes_t5, segments, keypoints, bbox_format='xyxy', normalized=False)
            # new_instances_t6 = Instances(bboxes_t6, segments, keypoints, bbox_format='xyxy', normalized=False)
            
            # Clip
            new_instances_t1.clip(*self.size)
            new_instances_t2.clip(*self.size)
            new_instances_t3.clip(*self.size)
            # new_instances_t4.clip(*self.size)
            # new_instances_t5.clip(*self.size)
            # new_instances_t6.clip(*self.size)

            # Filter instances
            instances_t1.scale(scale_w=scale, scale_h=scale, bbox_only=True)
            instances_t2.scale(scale_w=scale, scale_h=scale, bbox_only=True)
            instances_t3.scale(scale_w=scale, scale_h=scale, bbox_only=True)
            # instances_t4.scale(scale_w=scale, scale_h=scale, bbox_only=True)
            # instances_t5.scale(scale_w=scale, scale_h=scale, bbox_only=True)
            # instances_t6.scale(scale_w=scale, scale_h=scale, bbox_only=True)
            
            # Make the bboxes have the same scale with new_bboxes
            i_t1 = self.box_candidates(box1=instances_t1.bboxes.T,
                                    box2=new_instances_t1.bboxes.T,
                                    area_thr=0.01 if len(segments) else 0.10)
            i_t2 = self.box_candidates(box1=instances_t2.bboxes.T,
                                    box2=new_instances_t2.bboxes.T,
                                    area_thr=0.01 if len(segments) else 0.10)
            i_t3 = self.box_candidates(box1=instances_t3.bboxes.T,
                                    box2=new_instances_t3.bboxes.T,
                                    area_thr=0.01 if len(segments) else 0.10)
            # i_t4 = self.box_candidates(box1=instances_t4.bboxes.T,
            #                         box2=new_instances_t4.bboxes.T,
            #                         area_thr=0.01 if len(segments) else 0.10)
            # i_t5 = self.box_candidates(box1=instances_t5.bboxes.T,
            #                         box2=new_instances_t5.bboxes.T,
            #                         area_thr=0.01 if len(segments) else 0.10)
            # i_t6 = self.box_candidates(box1=instances_t6.bboxes.T,
            #                         box2=new_instances_t6.bboxes.T,
            #                         area_thr=0.01 if len(segments) else 0.10)
            i = [i_t1[num] and i_t2[num] and i_t3[num] for num in range(len(i_t1))]
            labels['instances_t1'] = new_instances_t1[i]
            labels['instances_t2'] = new_instances_t2[i]
            labels['instances_t3'] = new_instances_t3[i]
            # labels['instances_t4'] = new_instances_t4[i]
            # labels['instances_t5'] = new_instances_t5[i]
            # labels['instances_t6'] = new_instances_t6[i]
        else:    
            new_instances = Instances(bboxes, segments, keypoints, bbox_format='xyxy', normalized=False)
            # Clip
            new_instances.clip(*self.size)

            # Filter instances
            instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
            # Make the bboxes have the same scale with new_bboxes
            i = self.box_candidates(box1=instances.bboxes.T,
                                    box2=new_instances.bboxes.T,
                                    area_thr=0.01 if len(segments) else 0.10)
            labels['instances'] = new_instances[i]
        labels['cls'] = cls[i]
        labels['img'] = img
        labels['resized_shape'] = img.shape[:2]
        return labels

    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
        # Compute box candidates: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


class RandomHSV:

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels):
        """Applies random horizontal or vertical flip to an image with a given probability."""
        img = labels['img']
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            # Support for multi-channel images (e.g. cube mode with 9 channels)
            if img.shape[2] >= 3:
                for i in range(0, (img.shape[2] // 3) * 3, 3):
                    frame = img[:, :, i:i + 3]
                    hue, sat, val = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
                    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
                    img[:, :, i:i + 3] = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return labels


class RandomFlip:

    def __init__(self, p=0.5, direction='horizontal', flip_idx=None, cube=False) -> None:
        assert direction in ['horizontal', 'vertical'], f'Support direction `horizontal` or `vertical`, got {direction}'
        assert 0 <= p <= 1.0

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx
        self.cube = cube

    def __call__(self, labels):
        """Resize image and padding for detection, instance segmentation, pose."""
        img = labels['img']
        if self.cube:
            instances_t1 = labels.pop('instances_t1')
            instances_t2 = labels.pop('instances_t2')
            instances_t3 = labels.pop('instances_t3')
            # instances_t4 = labels.pop('instances_t4')
            # instances_t5 = labels.pop('instances_t5')
            # instances_t6 = labels.pop('instances_t6')

            instances_t1.convert_bbox(format='xywh')
            instances_t2.convert_bbox(format='xywh')
            instances_t3.convert_bbox(format='xywh')
            # instances_t4.convert_bbox(format='xywh')
            # instances_t5.convert_bbox(format='xywh')
            # instances_t6.convert_bbox(format='xywh')
            h, w = img.shape[:2]
            h = 1 if instances_t3.normalized else h
            w = 1 if instances_t3.normalized else w

            # Flip up-down
            if self.direction == 'vertical' and random.random() < self.p:
                img = np.flipud(img)
                instances_t1.flipud(h)
                instances_t2.flipud(h)
                instances_t3.flipud(h)
                # instances_t4.flipud(h)
                # instances_t5.flipud(h)
                # instances_t6.flipud(h)
            if self.direction == 'horizontal' and random.random() < self.p:
                img = np.fliplr(img)
                instances_t1.fliplr(w)
                instances_t2.fliplr(w)
                instances_t3.fliplr(w)
                # instances_t4.fliplr(w)
                # instances_t5.fliplr(w)
                # instances_t6.fliplr(w)
                # For keypoints
                if self.flip_idx is not None and instances_t3.keypoints is not None:
                    instances_t3.keypoints = np.ascontiguousarray(instances_t3.keypoints[:, self.flip_idx, :])
            labels['img'] = np.ascontiguousarray(img)
            labels['instances_t1'] = instances_t1
            labels['instances_t2'] = instances_t2
            labels['instances_t3'] = instances_t3
            # labels['instances_t4'] = instances_t4
            # labels['instances_t5'] = instances_t5
            # labels['instances_t6'] = instances_t6
        else:
            instances = labels.pop('instances')
            instances.convert_bbox(format='xywh')
            h, w = img.shape[:2]
            h = 1 if instances.normalized else h
            w = 1 if instances.normalized else w

            # Flip up-down
            if self.direction == 'vertical' and random.random() < self.p:
                img = np.flipud(img)
                instances.flipud(h)
            if self.direction == 'horizontal' and random.random() < self.p:
                img = np.fliplr(img)
                instances.fliplr(w)
                # For keypoints
                if self.flip_idx is not None and instances.keypoints is not None:
                    instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
            labels['img'] = np.ascontiguousarray(img)
            labels['instances'] = instances
        return labels


def custom_copyMakeBorder(src, top, bottom, left, right, borderType, value=0):
    """
    自定义实现类似 cv2.copyMakeBorder 的功能，可以处理任意形状的矩阵。
    
    参数:
        src: 输入矩阵，形状为 (H, W, C)。
        top: 在矩阵上方填充的像素行数。
        bottom: 在矩阵下方填充的像素行数。
        left: 在矩阵左侧填充的像素列数。
        right: 在矩阵右侧填充的像素列数。
        borderType: 边界填充方式，可选值为：
            - 'constant': 用指定的常数值填充边界。
            - 'replicate': 复制边缘像素。
            - 'reflect': 反射边缘像素，但不复制最边缘的像素。
        value: 当边界填充方式为 'constant' 时，指定填充的常数值。
    
    返回:
        扩展边界后的矩阵。
    """
    H, W, C = src.shape
    new_H = H + top + bottom
    new_W = W + left + right
    
    # 创建一个新矩阵，用于存放扩展边界后的结果
    if borderType == 'constant':
        dst = np.full((new_H, new_W, C), value, dtype=src.dtype)
    else:
        dst = np.zeros((new_H, new_W, C), dtype=src.dtype)
    
    # 将原始矩阵复制到新矩阵的中心位置
    dst[top:top+H, left:left+W] = src
    
    if borderType == 'replicate':
        # 复制边缘像素
        # 上边界
        dst[:top, left:left+W] = src[0, :, :][np.newaxis, ...]
        # 下边界
        dst[top+H:, left:left+W] = src[-1, :, :][np.newaxis, ...]
        # 左边界
        dst[top:top+H, :left] = src[:, 0, :][:, np.newaxis, :]
        # 右边界
        dst[top:top+H, left+W:] = src[:, -1, :][:, np.newaxis, :]
        # 左上角
        dst[:top, :left] = src[0, 0, :][np.newaxis, np.newaxis, :]
        # 右上角
        dst[:top, left+W:] = src[0, -1, :][np.newaxis, np.newaxis, :]
        # 左下角
        dst[top+H:, :left] = src[-1, 0, :][np.newaxis, np.newaxis, :]
        # 右下角
        dst[top+H:, left+W:] = src[-1, -1, :][np.newaxis, np.newaxis, :]
    
    elif borderType == 'reflect':
        # 反射边缘像素，但不复制最边缘的像素
        # 上边界
        dst[:top, left:left+W] = np.flip(src[:top, :, :], axis=0)
        # 下边界
        dst[top+H:, left:left+W] = np.flip(src[-bottom:, :, :], axis=0)
        # 左边界
        dst[top:top+H, :left] = np.flip(src[:, :left, :], axis=1)
        # 右边界
        dst[top:top+H, left+W:] = np.flip(src[:, -right:, :], axis=1)
        # 左上角
        dst[:top, :left] = np.flip(src[:top, :left, :], axis=(0, 1))
        # 右上角
        dst[:top, left+W:] = np.flip(src[:top, -right:, :], axis=(0, 1))
        # 左下角
        dst[top+H:, :left] = np.flip(src[-bottom:, :left, :], axis=(0, 1))
        # 右下角
        dst[top+H:, left+W:] = np.flip(src[-bottom:, -right:, :], axis=(0, 1))
    
    return dst


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32, cube=False):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.cube = cube

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'], (dw, dh))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        # print(f"DEBUG: img shape {img.shape}, top {top}, bottom {bottom}, left {left}, right {right}")
        if img.shape[2] <= 4:
            border_val = (114,) * img.shape[2]
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                     value=border_val)  # add border
        else:
            img = custom_copyMakeBorder(img, top, bottom, left, right, 'constant', value=114)
        # img = custom_copyMakeBorder(img, top, bottom, left, right, 'constant', 114)

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        if self.cube:
            labels['instances_t1'].convert_bbox(format='xyxy')
            labels['instances_t1'].denormalize(*labels['img'].shape[:2][::-1])
            labels['instances_t1'].scale(*ratio)
            labels['instances_t1'].add_padding(padw, padh)

            labels['instances_t2'].convert_bbox(format='xyxy')
            labels['instances_t2'].denormalize(*labels['img'].shape[:2][::-1])
            labels['instances_t2'].scale(*ratio)
            labels['instances_t2'].add_padding(padw, padh)

            labels['instances_t3'].convert_bbox(format='xyxy')
            labels['instances_t3'].denormalize(*labels['img'].shape[:2][::-1])
            labels['instances_t3'].scale(*ratio)
            labels['instances_t3'].add_padding(padw, padh)

            # labels['instances_t4'].convert_bbox(format='xyxy')
            # labels['instances_t4'].denormalize(*labels['img'].shape[:2][::-1])
            # labels['instances_t4'].scale(*ratio)
            # labels['instances_t4'].add_padding(padw, padh)

            # labels['instances_t5'].convert_bbox(format='xyxy')
            # labels['instances_t5'].denormalize(*labels['img'].shape[:2][::-1])
            # labels['instances_t5'].scale(*ratio)
            # labels['instances_t5'].add_padding(padw, padh)

            # labels['instances_t6'].convert_bbox(format='xyxy')
            # labels['instances_t6'].denormalize(*labels['img'].shape[:2][::-1])
            # labels['instances_t6'].scale(*ratio)
            # labels['instances_t6'].add_padding(padw, padh)
        else:
            labels['instances'].convert_bbox(format='xyxy')
            labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
            labels['instances'].scale(*ratio)
            labels['instances'].add_padding(padw, padh)
        return labels


class CopyPaste:

    def __init__(self, p=0.5, cube=False) -> None:
        self.p = p
        self.cube = cube

    def __call__(self, labels):
        """Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)."""
        im = labels['img']
        cls = labels['cls']
        h, w = im.shape[:2]
        if self.cube:
            instances_t1 = labels.pop('instances_t1')
            instances_t1.convert_bbox(format='xyxy')
            instances_t1.denormalize(w, h)

            instances_t2 = labels.pop('instances_t2')
            instances_t2.convert_bbox(format='xyxy')
            instances_t2.denormalize(w, h)

            instances_t3 = labels.pop('instances_t3')
            instances_t3.convert_bbox(format='xyxy')
            instances_t3.denormalize(w, h)

            # instances_t4 = labels.pop('instances_t4')
            # instances_t4.convert_bbox(format='xyxy')
            # instances_t4.denormalize(w, h)

            # instances_t5 = labels.pop('instances_t5')
            # instances_t5.convert_bbox(format='xyxy')
            # instances_t5.denormalize(w, h)

            # instances_t6 = labels.pop('instances_t6')
            # instances_t6.convert_bbox(format='xyxy')
            # instances_t6.denormalize(w, h)
        else:
            instances = labels.pop('instances')
            instances.convert_bbox(format='xyxy')
            instances.denormalize(w, h)
        if self.p and len(instances.segments):
            n = len(instances)
            _, w, _ = im.shape  # height, width, channels
            im_new = np.zeros(im.shape, np.uint8)

            # Calculate ioa first then select indexes randomly
            ins_flip = deepcopy(instances)
            ins_flip.fliplr(w)

            ioa = bbox_ioa(ins_flip.bboxes, instances.bboxes)  # intersection over area, (N, M)
            indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
            n = len(indexes)
            for j in random.sample(list(indexes), k=round(self.p * n)):
                cls = np.concatenate((cls, cls[[j]]), axis=0)
                instances = Instances.concatenate((instances, ins_flip[[j]]), axis=0)
                cv2.drawContours(im_new, instances.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)

            result = cv2.flip(im, 1)  # augment segments (flip left-right)
            i = cv2.flip(im_new, 1).astype(bool)
            im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

        labels['img'] = im
        labels['cls'] = cls
        if self.cube:
            labels['instances_t1'] = instances_t1
            labels['instances_t2'] = instances_t2
            labels['instances_t3'] = instances_t3
            # labels['instances_t4'] = instances_t4
            # labels['instances_t5'] = instances_t5
            # labels['instances_t6'] = instances_t6
        else:
            labels['instances'] = instances
        return labels


class Albumentations:
    # YOLOv8 Albumentations class (optional, only used if package is installed)
    def __init__(self, p=1.0, cube=False):
        """Initialize the transform object for YOLO bbox formatted params."""
        self.p = p
        self.transform = None
        self.cube = cube
        prefix = colorstr('albumentations: ')
        try:
            import albumentations as A

            check_version(A.__version__, '1.0.3', hard=True)  # version requirement

            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f'{prefix}{e}')

    def __call__(self, labels):
        """Generates object detections and returns a dictionary with detection results."""
        im = labels['img']
        cls = labels['cls']
        if len(cls):
            if self.cube:
                labels['instances_t1'].convert_bbox('xywh')
                labels['instances_t1'].normalize(*im.shape[:2][::-1])
                bboxes_t1 = labels['instances_t1'].bboxes

                labels['instances_t2'].convert_bbox('xywh')
                labels['instances_t2'].normalize(*im.shape[:2][::-1])
                bboxes_t2 = labels['instances_t2'].bboxes

                labels['instances_t3'].convert_bbox('xywh')
                labels['instances_t3'].normalize(*im.shape[:2][::-1])
                bboxes_t3 = labels['instances_t3'].bboxes

                # labels['instances_t4'].convert_bbox('xywh')
                # labels['instances_t4'].normalize(*im.shape[:2][::-1])
                # bboxes_t4 = labels['instances_t4'].bboxes

                # labels['instances_t5'].convert_bbox('xywh')
                # labels['instances_t5'].normalize(*im.shape[:2][::-1])
                # bboxes_t5 = labels['instances_t5'].bboxes

                # labels['instances_t6'].convert_bbox('xywh')
                # labels['instances_t6'].normalize(*im.shape[:2][::-1])
                # bboxes_t6 = labels['instances_t6'].bboxes
                # TODO: add supports of segments and keypoints
                if self.transform and random.random() < self.p:
                    new_t1 = self.transform(image=im, bboxes=bboxes_t1, class_labels=cls)  # transformed
                    if len(new_t1['class_labels']) > 0:  # skip update if no bbox in new im
                        labels['img'] = new_t1['image']
                        labels['cls'] = np.array(new_t1['class_labels'])
                        bboxes_t1 = np.array(new_t1['bboxes'], dtype=np.float32)
                labels['instances_t1'].update(bboxes=bboxes_t1)
                
                if self.transform and random.random() < self.p:
                    new_t2 = self.transform(image=im, bboxes=bboxes_t2, class_labels=cls)  # transformed
                    if len(new_t2['class_labels']) > 0:  # skip update if no bbox in new im
                        labels['img'] = new_t2['image']
                        labels['cls'] = np.array(new_t2['class_labels'])
                        bboxes_t2 = np.array(new_t2['bboxes'], dtype=np.float32)
                labels['instances_t2'].update(bboxes=bboxes_t2)

                if self.transform and random.random() < self.p:
                    new_t3 = self.transform(image=im, bboxes=bboxes_t3, class_labels=cls)  # transformed
                    if len(new_t3['class_labels']) > 0:  # skip update if no bbox in new im
                        labels['img'] = new_t3['image']
                        labels['cls'] = np.array(new_t3['class_labels'])
                        bboxes_t3 = np.array(new_t3['bboxes'], dtype=np.float32)
                labels['instances_t3'].update(bboxes=bboxes_t3)

                # if self.transform and random.random() < self.p:
                #     new_t4 = self.transform(image=im, bboxes=bboxes_t4, class_labels=cls)  # transformed
                #     if len(new_t4['class_labels']) > 0:  # skip update if no bbox in new im
                #         labels['img'] = new_t4['image']
                #         labels['cls'] = np.array(new_t4['class_labels'])
                #         bboxes_t4 = np.array(new_t4['bboxes'], dtype=np.float32)
                # labels['instances_t4'].update(bboxes=bboxes_t4)

                # if self.transform and random.random() < self.p:
                #     new_t5 = self.transform(image=im, bboxes=bboxes_t5, class_labels=cls)  # transformed
                #     if len(new_t5['class_labels']) > 0:  # skip update if no bbox in new im
                #         labels['img'] = new_t5['image']
                #         labels['cls'] = np.array(new_t5['class_labels'])
                #         bboxes_t5 = np.array(new_t5['bboxes'], dtype=np.float32)
                # labels['instances_t5'].update(bboxes=bboxes_t5)

                # if self.transform and random.random() < self.p:
                #     new_t6 = self.transform(image=im, bboxes=bboxes_t6, class_labels=cls)  # transformed
                #     if len(new_t6['class_labels']) > 0:  # skip update if no bbox in new im
                #         labels['img'] = new_t6['image']
                #         labels['cls'] = np.array(new_t6['class_labels'])
                #         bboxes_t6 = np.array(new_t6['bboxes'], dtype=np.float32)
                # labels['instances_t6'].update(bboxes=bboxes_t6)
            else:
                labels['instances'].convert_bbox('xywh')
                labels['instances'].normalize(*im.shape[:2][::-1])
                bboxes = labels['instances'].bboxes
                # TODO: add supports of segments and keypoints
                if self.transform and random.random() < self.p:
                    new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # transformed
                    if len(new['class_labels']) > 0:  # skip update if no bbox in new im
                        labels['img'] = new['image']
                        labels['cls'] = np.array(new['class_labels'])
                        bboxes = np.array(new['bboxes'], dtype=np.float32)
                labels['instances'].update(bboxes=bboxes)
        return labels


# TODO: technically this is not an augmentation, maybe we should put this to another files
class Format:

    def __init__(self,
                 bbox_format='xywh',
                 normalize=True,
                 return_mask=False,
                 return_keypoint=False,
                 mask_ratio=4,
                 mask_overlap=True,
                 batch_idx=True,
                 cube=False):
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask  # set False when training detection only
        self.return_keypoint = return_keypoint
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx  # keep the batch indexes
        self.cube = cube

    def __call__(self, labels):
        """Return formatted image, classes, bounding boxes & keypoints to be used by 'collate_fn'."""
        img = labels.pop('img')
        h, w = img.shape[:2]
        cls = labels.pop('cls')
        if self.cube:
            instances_t1 = labels.pop('instances_t1')
            instances_t1.convert_bbox(format=self.bbox_format)
            instances_t1.denormalize(w, h)

            instances_t2 = labels.pop('instances_t2')
            instances_t2.convert_bbox(format=self.bbox_format)
            instances_t2.denormalize(w, h)

            instances_t3 = labels.pop('instances_t3')
            instances_t3.convert_bbox(format=self.bbox_format)
            instances_t3.denormalize(w, h)

            # instances_t4 = labels.pop('instances_t4')
            # instances_t4.convert_bbox(format=self.bbox_format)
            # instances_t4.denormalize(w, h)

            # instances_t5 = labels.pop('instances_t5')
            # instances_t5.convert_bbox(format=self.bbox_format)
            # instances_t5.denormalize(w, h)

            # instances_t6 = labels.pop('instances_t6')
            # instances_t6.convert_bbox(format=self.bbox_format)
            # instances_t6.denormalize(w, h)
            
            nl = len(instances_t3)
        else:
            instances = labels.pop('instances')
            instances.convert_bbox(format=self.bbox_format)
            instances.denormalize(w, h)
            
            nl = len(instances)

        if self.return_mask:
            if nl:
                if self.cube:
                    masks, instances_t1, cls = self._format_segments(instances_t1, cls, w, h)
                    masks, instances_t2, cls = self._format_segments(instances_t2, cls, w, h)
                    masks, instances_t3, cls = self._format_segments(instances_t3, cls, w, h)
                    # masks, instances_t4, cls = self._format_segments(instances_t4, cls, w, h)
                    # masks, instances_t5, cls = self._format_segments(instances_t5, cls, w, h)
                    # masks, instances_t6, cls = self._format_segments(instances_t6, cls, w, h)
                else:
                    masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
            else:
                masks = torch.zeros(1 if self.mask_overlap else nl, img.shape[0] // self.mask_ratio,
                                    img.shape[1] // self.mask_ratio)
            labels['masks'] = masks
        if self.normalize:
            if self.cube:
                instances_t1.normalize(w, h)
                instances_t2.normalize(w, h)
                instances_t3.normalize(w, h)
                # instances_t4.normalize(w, h)
                # instances_t5.normalize(w, h)
                # instances_t6.normalize(w, h)
            else:
                instances.normalize(w, h)
        labels['img'] = self._format_img(img)
        labels['cls'] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        if self.cube:
            labels['bboxes_t1'] = torch.from_numpy(instances_t1.bboxes) if nl else torch.zeros((nl, 4))
            labels['bboxes_t2'] = torch.from_numpy(instances_t2.bboxes) if nl else torch.zeros((nl, 4))
            labels['bboxes_t3'] = torch.from_numpy(instances_t3.bboxes) if nl else torch.zeros((nl, 4))
            # labels['bboxes_t4'] = torch.from_numpy(instances_t4.bboxes) if nl else torch.zeros((nl, 4))
            # labels['bboxes_t5'] = torch.from_numpy(instances_t5.bboxes) if nl else torch.zeros((nl, 4))
            # labels['bboxes_t6'] = torch.from_numpy(instances_t6.bboxes) if nl else torch.zeros((nl, 4))
        else:
            labels['bboxes'] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        if self.return_keypoint:
            labels['keypoints'] = torch.from_numpy(instances.keypoints)
        # Then we can use collate_fn
        if self.batch_idx:
            labels['batch_idx'] = torch.zeros(nl)
        return labels

    def _format_img(self, img):
        """Format the image for YOLOv5 from Numpy array to PyTorch tensor."""
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1])
        img = torch.from_numpy(img)
        return img

    def _format_segments(self, instances, cls, w, h):
        """convert polygon points to bitmap."""
        segments = instances.segments
        if self.mask_overlap:
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)
            masks = masks[None]  # (640, 640) -> (1, 640, 640)
            instances = instances[sorted_idx]
            cls = cls[sorted_idx]
        else:
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)

        return masks, instances, cls


def v8_transforms(dataset, imgsz, hyp, stretch=False, cube=False):
    """Convert images to a size suitable for YOLOv8 training."""
    pre_transform = Compose([
        Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic, cube=cube),
        CopyPaste(p=hyp.copy_paste, cube=cube),
        RandomPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz), cube=cube),
            cube=cube,
        )])
    flip_idx = dataset.data.get('flip_idx', None)  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get('kpt_shape', None)
        if flip_idx is None and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f'data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}')

    return Compose([
        # pre_transform,
        MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup, cube=cube),
        Albumentations(p=1.0, cube=cube),
        RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(direction='vertical', p=hyp.flipud, cube=cube),
        RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx, cube=cube)])  # transforms


# Classification augmentations -----------------------------------------------------------------------------------------
def classify_transforms(size=224, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):  # IMAGENET_MEAN, IMAGENET_STD
    # Transforms to apply if albumentations not installed
    if not isinstance(size, int):
        raise TypeError(f'classify_transforms() size {size} must be integer, not (list, tuple)')
    if any(mean) or any(std):
        return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(mean, std, inplace=True)])
    else:
        return T.Compose([CenterCrop(size), ToTensor()])


def hsv2colorjitter(h, s, v):
    """Map HSV (hue, saturation, value) jitter into ColorJitter values (brightness, contrast, saturation, hue)"""
    return v, v, s, h


def classify_albumentations(
        augment=True,
        size=224,
        scale=(0.08, 1.0),
        hflip=0.5,
        vflip=0.0,
        hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,  # image HSV-Value augmentation (fraction)
        mean=(0.0, 0.0, 0.0),  # IMAGENET_MEAN
        std=(1.0, 1.0, 1.0),  # IMAGENET_STD
        auto_aug=False,
):
    # YOLOv8 classification Albumentations (optional, only used if package is installed)
    prefix = colorstr('albumentations: ')
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        check_version(A.__version__, '1.0.3', hard=True)  # version requirement
        if augment:  # Resize and crop
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale)]
            if auto_aug:
                # TODO: implement AugMix, AutoAug & RandAug in albumentations
                LOGGER.info(f'{prefix}auto augmentations are currently not supported')
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if any((hsv_h, hsv_s, hsv_v)):
                    T += [A.ColorJitter(*hsv2colorjitter(hsv_h, hsv_s, hsv_v))]  # brightness, contrast, saturation, hue
        else:  # Use fixed crop for eval set (reproducibility)
            T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size, width=size)]
        T += [A.Normalize(mean=mean, std=std), ToTensorV2()]  # Normalize and convert to Tensor
        LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        return A.Compose(T)

    except ImportError:  # package not installed, skip
        pass
    except Exception as e:
        LOGGER.info(f'{prefix}{e}')


class ClassifyLetterBox:
    # YOLOv8 LetterBox class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, size=(640, 640), auto=False, stride=32):
        """Resizes image and crops it to center with max dimensions 'h' and 'w'."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old
        h, w = round(imh * r), round(imw * r)  # resized image
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else self.h, self.w
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top:top + h, left:left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


class CenterCrop:
    # YOLOv8 CenterCrop class for image preprocessing, i.e. T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, size=640):
        """Converts an image from numpy array to PyTorch tensor."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top:top + m, left:left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ToTensor:
    # YOLOv8 ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, half=False):
        """Initialize YOLOv8 ToTensor object with optional half-precision support."""
        super().__init__()
        self.half = half

    def __call__(self, im):  # im = np.array HWC in BGR order
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
