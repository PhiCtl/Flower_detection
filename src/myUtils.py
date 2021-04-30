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


def label_reader(json_file, type='Flower'):
  """ Read a label file for an image in JSON format:
  Args: valid file path name
   type (str) flower for masks, and Flower for bbox
  Return: dictionnary of np.array dim:(Nx4) of bounding boxes coordinates [xmin, ymin, xmax, ymax]
  """
  data = pd.read_json(json_file)
  bboxes = {}
  
  for _, d in data.iterrows():
    name = d['External ID']

    pix = []
    for l in d['Label']['objects']:
        if 'bbox' in l:
            if l['title'] == type:
                b = l['bbox']
                pix.append([b['left'],b['top'], b['left']+b['width'], b['top']+b['height']])
        # if masks
    
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

def get_img_transformed(train=False): #TODO modify min and max sizes
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
  if train:
      transforms.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,hue=0.1))
      transforms.append(T.RandomErasing()) # to randomly erase some pixels (artifical occlusion)
      transforms.append(T.GaussianBlur(kernel_size=3)) # because high resolution pictures
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

def intersect(bbox_1, bbox_2):
  """
  Args:
    - bbox_1, bbox_2 : torch.Tensors of dim (4) (top left and bottom right corners of bounding box)
  Returns:
    - (bool) : if both boxes intersect
  """

  if bbox_1[1] > bbox_2[3]: # bbox 1 below bbox 2
    return False
  if bbox_1[3] < bbox_2[1]: # bbox 1 above bbox2
    return False
  if bbox_1[2] < bbox_2[0]: # bbox 1 left of bbox2
    return False
  if bbox_1[0] > bbox_2[2]: # bbox 1 right of bbox2
    return False
  return True


def IoU(bbox_1, bbox_2):
  
  """
  Args:
    - bbox_1, bbox_2 : torch.Tensors of dim (4) (top left and bottom right corners of bounding box)
  Returns:
    - (float) Intersection over Union (IoU)
  """

  if intersect(bbox_1, bbox_2):

    xm = max(bbox_1[0], bbox_2[0])
    ym = max(bbox_1[1], bbox_2[1])
    xM = min(bbox_1[2], bbox_2[2])
    yM = min(bbox_1[3], bbox_2[3])
    overlap_area = (xM - xm + 1) * (yM - ym + 1)
    
    tot_area = bbox_area(bbox_1) + bbox_area(bbox_2) - overlap_area
    
    return overlap_area / tot_area
   
  return 0

def write(dataset, prediction=None, gt=True):
    """
    Writes prediction or groundtruth bboxes to files
    :param dataset: (torch.Dataset)
    :param prediction: (dic) output of model
    :param gt: (bool) if we want to write groundtruth to files
   """
    os.chdir('/content/drive/MyDrive/GBH/results')

    if gt:
        os.chdir('groundtruths')
        # <class_name> <left> <top> <right> <bottom>
        for (_, target), i in zip(dataset, range(len(dataset))):
            name = dataset.imgs[i]
            file_name = name + '.txt'
            f = open(file_name,'w+') # open file in w mode
            for label, bbox in zip(target['labels'], target['boxes']):
                f.write("{} {} {} {} {}\r\n".format(label, bbox[0], bbox[1], bbox[2], bbox[3]))
            f.close()

    if prediction is not None :
        os.chdir('detections')
        # <class_name> <confidence> <left> <top> <right> <bottom>
        for pred, (_, target), i in zip(prediction, dataset, range(len(dataset))):
            name = dataset.imgs[i]
            file_name = name + '.txt'
            f = open(file_name, 'w+')
            for label, score, bbox in zip(pred['labels'], pred['scores'], pred['boxes']):
                f.write("{} {} {} {} {} {}\r\n".format(label, score, bbox[0], bbox[1], bbox[2], bbox[3]))
            f.close()

    os.chdir('/content')




