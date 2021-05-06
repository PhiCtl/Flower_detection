import pandas as pd
import cv2
import numpy as np
import torchvision.transforms as T
import os
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
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


def draw_bboxes(target, name, thresh = 0.8, img_dir = 'drive/MyDrive/GBH/data_test/images/'):
  # How it is used : draw_bboxes(prediction[0], name=dataset_test.imgs[target['image_id']],thresh=0.6)
    img_path = img_dir + name
    image = cv2.imread(img_path)
    for [xm,ym,xM,yM], label, score in zip(target["boxes"], target["labels"], target["scores"]):
      color = ()
      if label == 1: c = (255,0,0)
      if label == 2: c = (0,255,0)
      if score > thresh :
        image = cv2.rectangle(image, (xm,ym), (xM,yM), c, 2)
    
    rescale = (int(image.shape[1]/4), int(image.shape[0]/4))
    image = cv2.resize(image, rescale)
    cv2.imshow("Boundi boxes", image)

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
      #transforms.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,hue=0.1))
      #transforms.append(T.RandomErasing()) # to randomly erase some pixels (artifical occlusion)
      transforms.append(T.GaussianBlur(kernel_size=5, sigma=(0.01,2.0))) # because high resolution pictures
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

def write(dataset, prediction=None, gt=True, initial_dir = '/content/drive/MyDrive/GBH/results' ):
    """
    Writes prediction or groundtruth bboxes to files
    :param dataset: (torch.Dataset)
    :param prediction: (dic) output of model
    :param gt: (bool) if we want to write groundtruth to files
    :param initial_dir : should contain the two following subfolders: groundtruths and detections
   """
    os.chdir(initial_dir)

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
    
    
def eval_custom(dataset, model, device=device):
  "Should NOT be used for mask RCNN -> because too heavy in memory"

  predictions = []
  model.eval()

  with torch.no_grad():

    for img, _ in dataset:
      preds = model([img.to(device)])
      for p in preds:
        predictions.append(p)

  return predictions


def get_object_detection_model(num_classes, mtype = 'Resnet50_FPN'):
    # load a model pre-trained on COCO

    if mtype ==  'Resnet50_FPN':
      model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
      # get the number of input features for the classifier
      in_features = model.roi_heads.box_predictor.cls_score.in_features
      # replace the pre-trained head with a new one
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if mtype == 'MobileNetV3_largeFPN':

      model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
      # get the number of input features for the classifier
      in_features = model.roi_heads.box_predictor.cls_score.in_features
      # replace the pre-trained head with a new one
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if mtype == 'MobileNetV3_largeFPN_320':
      # Low resolution network
      model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
      # get the number of input features for the classifier
      in_features = model.roi_heads.box_predictor.cls_score.in_features
      # replace the pre-trained head with a new one
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if mtype == 'MaskRCNN':
      # load an instance segmentation model pre-trained on COCO
      model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

      # get the number of input features for the classifier
      in_features = model.roi_heads.box_predictor.cls_score.in_features
      # replace the pre-trained head with a new one
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

      # now get the number of input features for the mask classifier
      in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
      hidden_layer = 256
      # and replace the mask predictor with a new one
      model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                          hidden_layer,
                                                          num_classes)

    return model




