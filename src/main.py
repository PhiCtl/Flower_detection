from myClasses import*
from myUtils import*
from myTransforms import *
import torch, time
import matplotlib.pyplot as plt
from skspatial.objects import Points, Plane


########################################################################################################################
# LOAD MODEL AND DATA
########################################################################################################################
file_name = 'data_test/img1.jpg' #TODO real file_name
model_weights = 'models/.pt' #TODO real name

# device
cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")

# TODO: folder and files hierarchy (think about)
# -> models folder containing the best weights as follow : best_modelName.pt
# Prepare the image
img = cv2.imread(file_name)
transform = T.Compose([T.ToTensor(), T.Normalize(mean = MEAN_Imagenet, std=STD_Imagenet)])
img = transform(img)

# Get the detection model with corresponding weights
model = get_object_detection_model(2, 'MobileNetV3_largeFPN')
model.load_state_dict(torch.load(model_weights, map_location=device))
model.eval()
# Get the orientation model (or the detector)
# TODO -> find HSV range !!
# TODO -> compute matrix a affiner
########################################################################################################################
# MAKE PREDICTIONS
########################################################################################################################
pred = model([img.to(device)])
# TODO coordinates = pred_to_coordinates(pred, camera, mode='centroid', conf=0.5)
#  OR coordinates = fromMask2Coord(masks, camera)
# TODO masks = get_masks(bboxes, img)

# TODO opt plane = get_orientation(bboxes, img) with matrix

########################################################################################################################
# CLASSES
########################################################################################################################















