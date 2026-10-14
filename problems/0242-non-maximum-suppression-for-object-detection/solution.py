import numpy as np

def non_maximum_suppression(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression (NMS) to bounding boxes.
    
    Args:
        boxes: Array-like of shape (N, 4) with boxes in format [x1, y1, x2, y2]
        scores: Array-like of shape (N,) with confidence scores
        iou_threshold: float, IoU threshold for suppression (0 to 1)
    
    Returns:
        List of indices of kept boxes, ordered by descending score
        Returns -1 for invalid inputs
    """
    # Your code here

    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if boxes.size == 0 or scores.size == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    area = np.maximum(0, x2- x1) * np.maximum(0, y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        interect = w * h
        union = area[i] + area[rest] - interect

        iou = np.divide(
            interect,
            union,
            out=np.zeros_like(interect),
            where=union > 0
        )

        order = rest[iou <= iou_threshold]
    
    return keep
