import pandas as pd
import cv2
import numpy as np
import torchvision.transforms as T
import os
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torch.functional as F
#from google.colab.patches import cv2_imshow

MEAN_Imagenet = [0.485, 0.456, 0.406]
STD_Imagenet = [0.229, 0.224, 0.225]


def label_reader(json_file, type='flower'):
  """ Read a label file for an image in JSON format:
  Args: valid file path name
  Return: dictionnary of np.array dim:(Nx4) of bounding boxes coordinates [xmin, ymin, xmax, ymax]
  """
  data = pd.read_json(json_file)
  bboxes = {}
  
  for _, d in data.iterrows():
    name = d['External ID']

    pix = []
    for l in d['Label']['objects']:
        if l['value'] == type:
            if 'bbox' in l:
                b = l['bbox']
                pix.append([b['left'],b['top'], b['left']+b['width'], b['top']+b['height']])
    
    if pix:
        bboxes[name] = np.array(pix) # of size (N,4), with opencv image convention
  
  return bboxes


def draw_bboxes(image, bboxes):
    img = image.copy()
    for [xm,ym,xM,yM] in bboxes:
        img = cv2.rectangle(img, (xm,ym), (xM,yM), (255,0,0), 2)
    rescale = (int(img.shape[1]/4), int(img.shape[0]/4))
    img = cv2.resize(img, rescale)
    cv2.imshow("Image with bounding box", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2_imshow(img)

def get_img_transformed(): #TODO modify min and max sizes
  """
  Apply mandatory transforms on the image

  Returns:
            - Composition of transforms
  """
  transforms = []
  # converts the image into a PyTorch Tensor
  transforms.append(T.ToTensor())
  # image scaling and normalization
  transforms.append(T.Normalize(mean=MEAN_Imagenet, std=STD_Imagenet))
  return T.Compose(transforms)


def collate_double(batch):
    """
    collate function for the ObjectDetectionDataSet.
    Only used by the dataloader.
    from : https://johschmidt42.medium.com/train-your-own-object-detector-with-faster-rcnn-pytorch-8d3c759cfc70
    """
    x = [sample['image'] for sample in batch]
    y = [sample['target'] for sample in batch] #list of tensors
    return x, y


def draw_corners(image, corners):
  img = image.copy()
  for [a,b,c,d,e,f,g,h] in corners:
      img = cv2.circle(img, (a,b), radius=1, color=(255, 255, 255), thickness=2)
      img = cv2.circle(img, (c,d), radius=1, color=(255, 255, 255), thickness=2)
      img = cv2.circle(img, (e,f), radius=1, color=(255, 255, 255), thickness=2)
      img = cv2.circle(img, (g,h), radius=1, color=(255, 255, 255), thickness=2)
  #cv2_imshow(img)
  cv2.imshow("Image rotated with corners",img)

def bbox_area(bbox):
    """
    Compute bounding boxes area
    :param bboxes: (numpy array of dimensions (nb_boxes, 4)
    :return area: (numpy array of dimensions (nb_boxes,)

    """
    area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    return area