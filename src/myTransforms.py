import cv2
import numpy as np
from myUtils import draw_corners, bboxes_area


# from https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/
def get_corners(bboxes):
    """Get corners of bounding boxes
    
    Parameters
    ----------
    
    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape (N,4) where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format [x1, y1, x2, y2]
    
    returns
    -------
    
    numpy.ndarray
        Numpy array of shape (N,8) containing N bounding boxes each described by their 
        corner co-ordinates [x1, y1, x2, y2, x3, y3, x4, y4]      
        
    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def rotate(image, target, angle):
    """Rotate the image, the masks if any and the bounding boxes.
    
    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. Rotate the bounding boxes and return the closest square enclosing the items.
    
    Parameters
    ----------
    
    image : numpy.ndarray
        numpy image
    
    target : (dic)
        contains 'boxes': np.array, 'masks': optional 3d np.array of masks (2d np.array)
    
    angle : float
        angle by which the image is to be rotated
    
    Returns
    -------
    
    numpy.ndarray (nh, nw, 3)
        Rotated Image
    np.ndarray (N, nh, nw)
        Rotated Masks
    numpy.ndarray (N,4)
        Rotated bbox
    
    """

    # First parameters
    img = image.copy()
    w, h = img.shape[1], img.shape[0]
    cx, cy = w // 2, h // 2

    # Get rotation matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    # Rotate image
    img = cv2.warpAffine(image, M, (nW, nH))

    # Get corners and reshape them
    bboxes = target["boxes"].copy()
    corners = get_corners(bboxes)
    corners_ = corners.reshape(-1, 2)
    corners_ = np.hstack(
        (corners_, np.ones((corners_.shape[0], 1), dtype=type(corners_[0][0]))))  # to homogeneous coordinates

    # Rotate bounding box
    calculated = np.dot(M, corners_.T).T
    corners[:, :8] = calculated.reshape(-1, 8)

    # Get closest enclosing box
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    # draw_corners(img, corners)

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)
    new_bbox = np.hstack((xmin, ymax, xmax, ymin))

    # Rescale everybody
    scale_x = img.shape[1] / w
    scale_y = img.shape[0] / h
    img = cv2.resize(img, (w, h))

    # scale and clip bounding box
    new_bbox = np.divide(new_bbox, np.array([scale_x, scale_y, scale_x, scale_y])).astype(int)  # rescale
    new_bbox[:, 0] = np.maximum(new_bbox[:, 0], 0)
    new_bbox[:, 1] = np.maximum(new_bbox[:, 1], 0)
    new_bbox[:, 2] = np.minimum(new_bbox[:, 2], w)
    new_bbox[:, 3] = np.minimum(new_bbox[:, 3], h)

    new_target = {'boxes': new_bbox}

    # Rotate and resize masks, if any
    if 'masks' in target:
        masks = target['masks'].copy()
        for mask in masks:
            mask = cv2.warpAffine(mask, M, (nW, nH))
            mask = cv2.resize(mask, (w, h))
        new_target['masks'] = masks

    return img, new_target
