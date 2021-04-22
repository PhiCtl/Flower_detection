from src.myClasses import*

myTransforms = RandomRotate(90)
dataset = FlowerDetectionDataset('data_train/', 'labels/export1.json', customed_transforms=myTransforms)

dic = dataset[0]