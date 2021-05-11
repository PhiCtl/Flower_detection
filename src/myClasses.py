import torch
import torchvision
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
    


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
      """
      Args: output size should be of the following form 
        - int: for square pictures
        - tuple = (new height, new width): otherwise
      """
      assert isinstance(output_size, (int, tuple))
      self.output_size = output_size

    def __call__(self, img, tgt=None):
        image = img.copy()

        # reshape image
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image = cv2.resize(image, (new_h, new_w))

        if tgt is not None:
          target = tgt.copy()
          # resizing bounding boxes
          boxes = target['boxes']
          scale_x = w / new_w
          scale_y = h / new_h

          # rescale and clip
          new_bbox = np.divide(boxes, np.array([scale_x, scale_y, scale_x, scale_y])).astype(int)  # rescale
          target['boxes'] = new_bbox

          return image, target
        else:
          return image

class FlowerDetectionDataset(torch.utils.data.Dataset):

    def __init__(self, root_img, json_file_root, core=False, transforms=None, custom_transforms=None):
        self.root_img = root_img
        self.transforms = transforms
        self.custom_transforms = custom_transforms
        self.include_core = core

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(root_img)))  # OK
        if '.ipynb_checkpoints' in self.imgs: self.imgs.remove('.ipynb_checkpoints')
        if self.include_core :
            self.core_labels = label_reader(json_file_root, type='Core')
        else:
            self.flower_labels = label_reader(json_file_root)
        


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):

        # load images
        img_path = os.path.join(self.root_img, self.imgs[idx])
        img = cv2.imread(img_path)  # read in [H,W,3] BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # turn to RGB [H,W,3]

        # load bounding boxes
        if self.include_core and self.imgs[idx] in self.core_labels:
            bboxes = np.array(self.core_labels[self.imgs[idx]])
        else:
            bboxes = np.array(self.flower_labels[self.imgs[idx]])

        # Apply transforms #TODO all in once, clean up ugly code
        # Still img: np.array and bboxes:np.array
        if self.custom_transforms is not None:  # include bboxes transforms
            img, res = self.custom_transforms(img, {'boxes': bboxes})  # img still in cv2 convention
            bboxes = res['boxes']

        # there is only one class (either background either flower/core)
        labels = torch.ones((len(bboxes),), dtype=torch.int64)
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

    def __init__(self, root_img, root_masks, transforms=None, custom_transforms=None):
        self.root_img = root_img

        self.root_masks = root_masks
        self.transforms = transforms
        self.custom_transforms = custom_transforms

        # load all image files names and masks folder names, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(root_img)))  # OK
        if '.ipynb_checkpoints' in self.imgs: self.imgs.remove('.ipynb_checkpoints')
        self.masks = list(sorted(os.listdir(root_masks))) # list of folders with img name
        if '.ipynb_checkpoints' in self.masks: self.masks.remove('.ipynb_checkpoints')

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
            mask_img = cv2.imread(path,0)/255

            # Build bboxes
            pos = np.where(mask_img)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            bboxes.append([xmin, ymin, xmax, ymax])
            masks.append(mask_img)
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
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.ones((len(bboxes),), dtype=torch.int64)
        area = torch.abs((bboxes[:, 2] - bboxes[:, 0])*(bboxes[:, 3] - bboxes[:, 1]))
        iscrowd = torch.zeros((len(bboxes),), dtype=torch.int64) # all instances are not crowd (?!) # TODO what is iscrowd

        # Prepare sample
        target = {"boxes": bboxes,\
                  "masks": torch.as_tensor(masks, dtype=torch.uint8), "image_id": torch.tensor([idx]), "labels": labels, 'iscrowd': iscrowd, 'area': area}


        return img, target

