from myClasses import*

myTransforms = RandomRotate(90)
dataset = FlowerDetectionDataset(root_img='data_train/', json_file_root='labels/export1.json', custom_transforms=myTransforms)

dic = dataset[0]