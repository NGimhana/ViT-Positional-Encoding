import torchvision.datasets as datasets
import torchvision.transforms as transforms
    
# Define the path to save the downloaded data
data_path = '/home/kgdesilva/Desktop/HPC/Project/SemanticSegmentation/segmenter/data/coco'

# Download the training set
train_set = datasets.CocoDetection(root=data_path, annFile=f'{data_path}/annotations/instances_train2017.json', transform=transforms.ToTensor())

# Download the validation set
val_set = datasets.CocoDetection(root=data_path, annFile=f'{data_path}/annotations/instances_val2017.json', transform=transforms.ToTensor())    
    