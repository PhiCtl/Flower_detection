import numpy as np
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import torch.functional as F
import cv2
from myUtils import*
from myClasses import *

# ref https://github.com/pytorch/vision/blob/master/references/detection/
# ref https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

################################################################################
# GETTING STARTED
################################################################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

myTransforms = RandomRotate(90)
transforms_train = get_img_transformed(train=True)
transforms_test = get_img_transformed(train=False)

################################################################################
# LOAD DATASET
# data_train/
# data_test/
#          
################################################################################
dataset = FlowerDetectionDataset('data_train/', 'labels/export1.json', customed_transforms=myTransforms, transforms=transforms_train)
data_loader = DataLoader(
    dataset, batch_size=12, shuffle=True, num_workers=2,
    collate_fn=collate_double)

dataset_test = FlowerDetectionDataset('data_tes/', 'labels/export2.json', transforms=transforms_test)
data_loader_test = DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=2,
    collate_fn=collate_double)

################################################################################
# TRAINING
################################################################################
# define model
model = myModel()
# move model to device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
# let's train it for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)   

                                          