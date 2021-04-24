from src.myClasses import*

myTransforms = RandomRotate(90)
dataset = FlowerDetectionDataset('data_train/', '../labels/export1.json', custom_transforms=myTransforms)

dic = dataset[0]