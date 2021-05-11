def metrics(prediction, groundtruth, iou_thresh, num_classes = 3):
  """
  thanks to https://github.com/johschmidt42/PyTorch-Object-Detection-Faster-RCNN-Tutorial/metrics/pascal_voc_evaluator.py
  Args:
    - prediction: (dic) predictions (boxes, labels, scores)
    - groundtruth: (dic) groundtruth (bboxes, labels, name, image_id,...)
    - num_classes: (int) number of classes + 1 (for background)
  Returns: per class and in total
    - in total (dic) : perclass (bool = False), Precision, Recall
    - per class (dic) : perclass (bool = True), Precision, Recall, TP, FP, FN
  """

TP = np.zeros((num_classes-1,))
FP = np.zeros((num_classes-1,))
FN = np.zeros((num_classes-1,))

# find number of detected versus gt instances number
det_nb = len(prediction['boxes'])
actual_nb = len(groundtruth['boxes'])

# keep track of already matched detections              
detected_gt_match = np.zeros((actual_nb, )))

# predictions already sorted by decreasing score (confidence)

# Compute TP, FP, FN
for pred, label in zip(prediction['boxes'], prediction['labels']):
  
  iou_max = 0
  gt_label = 0
  ind = 0

  # find gt bbox of max IoU
  for gt, gt_lab, j in zip(groundtruth['boxes'], groundtruth['labels'], range(actual_nb)):
    iou = IoU(pred, gt)
    # update max
    if iou > iou_max:
      iou_max = iou
      gt_label = gt_lab
      ind = j
  
  # Set TP, TN, FP or FN
  # If we already have a prediction matching the found gt prediction, then it is a false positive or a FP and FN
  # Else, we enquire about the predicted label
  if detected_gt_match[ind] == 0:

    # If we predicted the wrong class, then this is a false positive for wrong class
    # and a false negative for true class
    if label != gt_label:
      FP[label] += 1
      FN[gt_label] +=1

    # If labels match, then we have a correct prediction
    if label == gt_label:
      detected_gt_match[ind] = 1 # we update the detection match array
      TP[gt_label] += 1 # we update the number of TP for this class

  # So if we already have a matching prediction for this given gt bbox, this is a FP or a FP and FN
  else :
    # if both labels are different, then this is a false positive for wrong class
    # and a false negative for true class
    if label != gt_label:
      FP[label] += 1
      FN[gt_label] +=1
    # else, this is a false positive for (though) correctly predicted class
    else:
      FP[label] +=1
    
# Compute precision and recall per class and in total
precision = TP.sum() / (TP.sum() + FP.sum()) # proportion of correct detections
recall = TP.sum() / (TP.sum() + FN.sum()) # proportion of detected elements
precision_perclass = TP / (TP + FP)
recall_perclass = TP / (TP + FN)

return {'perclass': False, 'precision': precision, 'recall': recall},\
       {'perclass': True, 'precision': precision_perclass, 'recall':recall_perclass, 'TP': TP, 'FP': FP, 'FN': FN}

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

  if intersect(bbox_1, bbox_2):

    xm = max(bbox_1[0], bbox_2[0])
    ym = max(bbox_1[1], bbox_2[1])
    xM = min(bbox_1[2], bbox_2[2])
    yM = min(bbox_1[3], bbox_2[3])
    overlap_area = (xM - xm + 1) * (yM - ym + 1)
    
    tot_area = bboxes_area(bbox_1) + bboxes_area(bbox_2) - overlap_area
    
    return overlap_area / tot_area
  return 0



import pytorch_lightning as pl

class FasterRCNN_lightning(pl.LightningModule):

  def __init__(self,
               model: torch.nn.Module,
               lr: float = 0.0001,
               iou_threshold: float 0 0.5):
    super().__init__()


    self.model = model
    self.num_classes = self.model.num_classes
    self.lr = lr
    self.iou_threshold=iou_threshold

    # Transformation parameters
    self.mean = model.image_mean
    self.std = model.image_std
    self.min_size = model.min_size
    self.max_size = model.max_size

    self.save_hyperparameters()

  def forward(self, x):
    self.model.eval()
    return self.model(x)

  def training_step(self, batch, batch_idx):
    x, y, x_name, y_name = batch # unpacking of the tuple returned by batch

    loss_dict = self.model(x,y)
    loss = sum(loss for loss in loss_dict.values())
    self.log_dict(loss_dict)
    return loss

  def validation_step(self, batch, batch_idx):

    x,y,x_name,y_name = batch

    # Inference part
    predictions = self.model(x) # is a list of dict {boxes, labels, scores, masks if any} 
    gt_boxes =  [build_bboxes_list(target, name) for target, name in zip(y, y_name)] # returns a list of lists of  BoundingBox objects
    pred_boxes = [build_bboxes_list(target, y_name) for target, name in zip(predictions, y_name)] #returns a list of lists of BoundingBox objects

    return {'gt_boxes': gt_boxes, 'pred_boxes': pred_boxes}
  
  def validation_epoch_end(self, outs):

    gt_boxes = [out['gt_boxes'] for out in outs]
    pred_boxes = [out['pred_boxes'] for out in outs]

    from metrics.pascal_voc_evaluator import get_pascalvoc_metrics
    from metrics.enumerators import MethodAveragePrecision
    metric = get_pascalvoc_metrics(gt_boxes=gt_boxes,
                                       det_boxes=pred_boxes,
                                       iou_threshold=self.iou_threshold,
                                       method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                       generate_table=True)

    per_class, mAP = metric['per_class'], metric['mAP']
    self.log('Validation_mAP', mAP)

    for key, value in per_class.items():
    self.log(f'Validation_AP_{key}', value['AP'])
    
    
    
class BoundingBox():
  """ Goal of this class: from neural network predictions, build convenient bounding boxes for evaluation
  Param:
    - ground_truth : (bool): if ground truth bounding box, we don't have a score #TODO is it useful ?
    - name : (string) corresponding image name
    - bbox : (torch.Tensor) bounding box
    - label : (int) object class
    - score : (float) confidence
  Builds a bbox object
  """

  def __init__(self, 
               name: str, 
               bbox: torch.Tensor,
               label: int,
               score):
    self.name = name
    self.bbox = bbox
    self.label = label
    self.score = score

  def __gt__(self, other):
    return self.score > other.score

  def __lt__(self, other):
    return self.score < other.score

  def get_area(self):
    return ((bbox[0] - bbox[2])*(bbox[1]-bbox[3])).abs()


def build_bboxes_list(target, name):
  """
  Computes bounding boxes list for a single prediction
  Param:
    - target (dic): 'boxes', 'labels', 'scores', 'image_id' #TODO redundancy with name to correct
    - name (string)
  """
  bb_list = []
  for box, label, score in zip(target['boxes'], target['labels'], target['scores']):
    bb_list.append(BoundingBox(name, box, label, score))

  return bb_list
  



