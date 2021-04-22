import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from myUtils import*
from myTransforms import*
import cv2
import os


class RandomRotate(object):
    """Apply randomly an rotation transformation on the image and the corresponding bounding boxes.

    Args: degrees (min, max)
        
    """

    def __init__(self, degrees):
      self.angle = np.random.uniform(-degrees, degrees)

        
    def __call__(self, image, bboxes):

        # Apply rotation 
        img, boxes = rotate(image, bboxes, self.angle)
        draw_bboxes(img, boxes)
        return img, boxes


class FlowerDetectionDataset(torch.utils.data.Dataset):

  def __init__(self, root, json_file_root, transforms=None, customed_transforms=None):
    self.root = root
    self.transforms = transforms
    self.customed_transforms = customed_transforms
    # load all image files, sorting them to
    # ensure that they are aligned
    self.imgs = list(sorted(os.listdir(root))) # OK
    self.labels = label_reader(json_file_root) #TODO test                                    

  def __len__(self):
    return len(self.imgs)
  
  def __getitem__(self, idx: int):

     # load images and bounding boxes
     img_path = os.path.join(self.root, self.imgs[idx])
     img = cv2.imread(img_path) # read in [H,W,3] BGR format
     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # turn to RGB [H,W,3]
     bboxes = np.array(self.labels[self.imgs[idx]]) # retrieve in dictionnary associated np.array

     # there is only one class (either background either flower)
     labels = torch.ones((len(bboxes),), dtype=torch.int64)
    
     # Apply transforms
     if self.customed_transforms is not None: # include bboxes transforms
       img, bboxes = self.customed_transforms(img, bboxes) # img still in cv2 convention
     if self.transforms is not None: # img transforms only
       img = self.transforms(img) # img toTensor

    # Prepare sample
     target= {}
     target["boxes"] = torch.as_tensor(bboxes, dtype=torch.float32)
     image_id = torch.tensor([idx])
     target["labels"] = labels
     target["image_id"] = image_id
     
     return {'x': img, 'y': target, 'x_name': self.imgs[idx] , 'y_name': self.imgs[idx]}

class myModel(torch.nn.Module):

  def __init__(self, model_type = 'fasterrcnn_resnet50_fpn', num_classes = 2):
    super().__init__()
    
    self.num_classes = num_classes
    self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  def forward(self,x):
    return self.model(x)

