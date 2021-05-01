import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from myUtils import*
from myTransforms import *
import cv2
import os


class RandomRotate(object):
    """Apply randomly an rotation transformation on the image, the corresponding masks and the corresponding bounding boxes.

    Args: degree
        
    """

    def __init__(self, degrees):
        self.angle = np.random.uniform(-degrees, degrees)

    def __call__(self, image, target):
        # Apply rotation
        img, targ = rotate(image, target, self.angle)
        #draw_bboxes(img, targ['boxes'])
        return img, targ

class FlowerDetectionDataset(torch.utils.data.Dataset):

    def __init__(self, root_img, json_file_root=None, transforms=None, custom_transforms=None):
        self.root_img = root_img
        self.transforms = transforms
        self.custom_transforms = custom_transforms

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(root_img)))  # OK
        self.flower_labels = label_reader(json_file_root)
        self.hidden_labels = label_reader(json_file_root, type='hidden')
        self.core_labels = label_reader(json_file_root, type='core')


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):

        # load images
        img_path = os.path.join(self.root_img, self.imgs[idx])
        img = cv2.imread(img_path)  # read in [H,W,3] BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # turn to RGB [H,W,3]

        # load bounding boxes
        bboxes = np.array(self.flower_labels[self.imgs[idx]])
        nb_flowers = len(bboxes)
        if self.imgs[idx] in self.hidden_labels:
           bboxes = np.vstack((bboxes, np.array(self.hidden_labels[self.imgs[idx]])))
        #if self.imgs[idx] in self.core_labels:
            #bboxes = np.vstack((bboxes, np.array(self.core_labels[self.imgs[idx]])))

        # Apply transforms #TODO all in once, clean up ugly code
        # Still img: np.array and bboxes:np.array
        if self.custom_transforms is not None:  # include bboxes transforms
            img, res = self.custom_transforms(img, {'boxes': bboxes})  # img still in cv2 convention
            bboxes = res['boxes']

        # there is only one class (either background either flower)
        labels = torch.ones((len(bboxes),), dtype=torch.int64)
        #labels[nb_flowers:] *= 2 # assign class 2 to flower cores
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        area = torch.abs(bboxes[:, 2] - bboxes[:, 0])*(bboxes[:, 1] - bboxes[:, 3])
        iscrowd = torch.zeros((len(bboxes),), dtype=torch.int64) # all instances are not crowd (?!) # TODO what is iscrowd

        # Prepare sample
        target = {"boxes": bboxes, "image_id": torch.tensor([idx]),\
                  "labels": labels, 'iscrowd': iscrowd, 'area': area}

        if self.transforms is not None:  # img transforms only
            img = self.transforms(img)  # img toTensor

        return img, target



class FlowerMaskDetectionDataset(torch.utils.data.Dataset):

    def __init__(self, root_img, root_masks=None, transforms=None, custom_transforms=None):
        self.root_img = root_img

        self.root_masks = root_masks
        self.transforms = transforms
        self.custom_transforms = custom_transforms

        # load all image files names and masks folder names, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(root_img)))  # OK
        self.imgs.remove('masks')
        self.imgs.remove('.ipynb_checkpoints')
        self.masks = list(sorted(os.listdir(root_masks))) # list of folders with img name


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):

        # load images
        img_path = os.path.join(self.root_img, self.imgs[idx])
        img = cv2.imread(img_path)  # read in [H,W,3] BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # turn to RGB [H,W,3]

        # load masks and set bboxes
        mask_path = os.path.join(self.root_masks, self.masks[idx]) # image corresponding mask folder
        masks_list = list(sorted(os.listdir(mask_path))) # list all masks
        masks, bboxes = [], []
        for mask in masks_list:
            path = os.path.join(mask_path, mask)
            mask_img = cv2.imread(path,0)
            masks.append(mask_img)

            # Build bboxes
            pos = np.where(mask_img)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            bboxes.append([xmin, ymin, xmax, ymax])
        masks = np.array(masks)
        bboxes = np.array(bboxes)


        # Apply transforms #TODO all in once, clean up ugly code
        target = {}
        if self.custom_transforms is not None:  # include bboxes transforms
            img, res = self.custom_transforms(img, {'boxes': bboxes})  # img still in cv2 convention
            bboxes = res['boxes']
        if self.transforms is not None:  # img transforms only
            img = self.transforms(img)  # img toTensor

        # there is only one class (either background either flower)
        labels = torch.ones((len(bboxes),), dtype=torch.int64)
        area = torch.abs((bboxes[:, 2] - bboxes[:, 0])*(bboxes[:, 3] - bboxes[:, 1]))
        iscrowd = torch.zeros((len(bboxes),), dtype=torch.int64) # all instances are not crowd (?!) # TODO what is iscrowd

        # Prepare sample
        target = {"boxes": torch.as_tensor(bboxes, dtype=torch.float32),\
                  "masks": torch.as_tensor(masks, dtype=torch.uint8), "image_id": torch.tensor([idx]), "flower_labels": labels, 'iscrowd': iscrowd, 'area': area}


        return img, target


class myModel(torch.nn.Module):

    def __init__(self, model_type='fasterrcnn_resnet50_fpn', num_classes=2):
        super().__init__()

        self.num_classes = num_classes
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
