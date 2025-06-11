# argus_track/utils/iou.py

"""IoU (Intersection over Union) utilities for tracking"""

from typing import List, Union

import numpy as np
from numba import jit

from ..core import Detection, Track


@jit(nopython=True)
def calculate_iou_jit(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate IoU between two bounding boxes (numba accelerated)

    Args:
        bbox1: First bbox in [x1, y1, x2, y2] format
        bbox2: Second bbox in [x1, y1, x2, y2] format

    Returns:
        IoU value between 0 and 1
    """
    # Get intersection coordinates
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate IoU between two bounding boxes

    Args:
        bbox1: First bbox in [x1, y1, x2, y2] format
        bbox2: Second bbox in [x1, y1, x2, y2] format

    Returns:
        IoU value between 0 and 1
    """
    return calculate_iou_jit(bbox1, bbox2)


@jit(nopython=True)
def calculate_iou_matrix_jit(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU matrix between two sets of bounding boxes (numba accelerated)

    Args:
        bboxes1: First set of bboxes in [N, 4] format
        bboxes2: Second set of bboxes in [M, 4] format

    Returns:
        IoU matrix of shape [N, M]
    """
    n_bbox1 = bboxes1.shape[0]
    n_bbox2 = bboxes2.shape[0]
    iou_matrix = np.zeros((n_bbox1, n_bbox2))

    for i in range(n_bbox1):
        for j in range(n_bbox2):
            iou_matrix[i, j] = calculate_iou_jit(bboxes1[i], bboxes2[j])

    return iou_matrix


def calculate_iou_matrix(
    tracks_or_bboxes1: Union[List[Track], np.ndarray],
    detections_or_bboxes2: Union[List[Detection], np.ndarray],
) -> np.ndarray:
    """
    Calculate IoU matrix between tracks and detections

    Args:
        tracks_or_bboxes1: List of tracks or array of bboxes
        detections_or_bboxes2: List of detections or array of bboxes

    Returns:
        IoU matrix of shape (len(tracks_or_bboxes1), len(detections_or_bboxes2))
    """
    # Handle different input types
    if isinstance(tracks_or_bboxes1, np.ndarray):
        bboxes1 = tracks_or_bboxes1
    else:
        bboxes1 = np.array([track.to_tlbr() for track in tracks_or_bboxes1])

    if isinstance(detections_or_bboxes2, np.ndarray):
        bboxes2 = detections_or_bboxes2
    else:
        bboxes2 = np.array([det.tlbr for det in detections_or_bboxes2])

    # Calculate IoU matrix
    return calculate_iou_matrix_jit(bboxes1, bboxes2)
