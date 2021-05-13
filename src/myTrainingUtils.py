import torch, torchvision, cv2

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



def eval_custom(dataset, model, device):
    "Should NOT be used for mask RCNN -> because too heavy in memory"

    predictions = []
    model.eval()

    with torch.no_grad():

        for img, _ in dataset:
            preds = model([img.to(device)])
            for p in preds:
                predictions.append(p)

    return predictions


def eval_custom_YOLO(dataset, model):
    """Custom evaluation for YOLO models only"""

    predictions = []
    model.eval()

    with torch.no_grad():
        for i in range(len(dataset)):
            path = '/content/drive/MyDrive/GBH/data_test/images/' + dataset.imgs[i]
            img = cv2.imread(path)
            #img = cv2.resize(img, (640, 640))
            results = model(img)
            target = {}
            target['boxes'] = results.xyxy[0][:, :4]
            target['scores'] = results.xyxy[0][:, 4].flatten()
            target['labels'] = target['scores'] * 0 + 1
            predictions.append(target)

    return predictions


def get_object_detection_model(num_classes, mtype='Resnet50_FPN'):
    # load a model pre-trained on COCO

    if mtype == 'Resnet50_FPN':
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

    if mtype == 'YOLOv5x':
        model = torch.hub.load('ultralytics/yolov5', 'custom',
                               path='/content/drive/MyDrive/GBH/models/yolo_07052021_1710/best.pt')

    return model

