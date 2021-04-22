import pandas as pd
import cv2
import numpy as np
import torchvision.transforms as T
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torch.functional as F
from google.colab.patches import cv2_imshow

MEAN_Imagenet = [0.485, 0.456, 0.406]
STD_Imagenet = [0.229, 0.224, 0.225]


def label_reader(json_file):
  """ Read a label file for an image in JSON format:
  Args: valid file path name
  Return: dictionnary of np.array dim:(Nx4) of bounding boxes coordinates [xmin, ymin, xmax, ymax]
  """
  data = pd.read_json(json_file)
  bboxes = {}
  
  for _, d in data.iterrows():
    name = d['External ID']
    labels = d['Label']['objects']
    pix = []
    
    for l in labels:
      b = l['bbox']
      pix.append([b['left'],b['top'], b['left']+b['width'], b['top']+b['height']])
    
    bboxes[name] = np.array(pix) # of size (N,4), with opencv image convention
  
  return bboxes


def draw_bboxes(image, bboxes):
    img = image.copy()
    for [xm,ym,xM,yM] in bboxes:
      img = cv2.rectangle(img, (xm,ym), (xM,yM) ,(255,0,0),2)
    #cv2.imshow("Image with bounding box", img) 
    cv2_imshow(img)

def get_img_transformed(train, min_size=2448, max_size=2448): #TODO modify min and max sizes
  """
  Apply mandatory transforms on the image
  Parameters:
            - train (bool): True if training, False if evaluating
  Returns:
            - Composition of transforms
  """
  transforms = []
  # converts the image into a PyTorch Tensor
  transforms.append(T.ToTensor())
  # image scaling and normalization
  transforms.append(GeneralizedRCNNTransform(min_size=min_size,
                                     max_size=max_size,
                                     image_mean=MEAN_Imagenet,
                                     image_std=STD_Imagenet))
  if train:
      # during training, randomly flip the training images
      # and ground-truth for data augmentation
      transforms.append(T.ColorJitter(brightness = 0.7, hue=0.2))
      transforms.append(T.RandomErasing())
      
  return T.Compose(transforms)


  def collate_double(batch):
    """
    collate function for the ObjectDetectionDataSet.
    Only used by the dataloader.
    from : https://johschmidt42.medium.com/train-your-own-object-detector-with-faster-rcnn-pytorch-8d3c759cfc70
    """
    x = [sample['x'] for sample in batch]
    y = [sample['y'] for sample in batch] #list of tensors
    x_name = [sample['x_name'] for sample in batch]
    y_name = [sample['y_name'] for sample in batch]
    return x, y, x_name, y_name


def draw_corners(image, corners):
  img = image.copy()
  for [a,b,c,d,e,f,g,h] in corners:
      img = cv2.circle(img, (a,b), radius=1, color=(255, 255, 255), thickness=2)
      img = cv2.circle(img, (c,d), radius=1, color=(255, 255, 255), thickness=2)
      img = cv2.circle(img, (e,f), radius=1, color=(255, 255, 255), thickness=2)
      img = cv2.circle(img, (g,h), radius=1, color=(255, 255, 255), thickness=2)
  cv2_imshow(img)

def bboxes_area(bboxes):
  area = np.abs(np.dot(bboxes[:,0] - bboxes[:,2], bboxes[:,1] - bboxes[:,3]))
  return area 
