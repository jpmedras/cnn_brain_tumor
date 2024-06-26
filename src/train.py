import numpy as np
from torch import manual_seed
import random
from torchvision.transforms import Compose, ToTensor, RandomRotation, RandomCrop, Resize, ElasticTransform, ColorJitter
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch import load
from torch import zeros
import matplotlib.pyplot as plt
from alexnet_pretrained import AlexNet

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set seeds
np.random.seed(42)
manual_seed(42)
random.seed(42)

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        y_ = zeros(2)
        y_[y] = 1
        if self.transform:
            x = self.transform(x)
        return x, y_

    def __len__(self):
        return len(self.dataset)
    
def data_augmentation(subset, transforms):
    datasets = [CustomDataset(subset)]

    for transform in transforms:
        datasets.append(CustomDataset(subset, transform))

    return ConcatDataset(datasets)

def plot_img(image, label=None):
    plt.imshow(image.permute((1, 2, 0)), cmap='gray')
    plt.axis('off')
    if label is None :
        plt.title('Image in Tensor format')
    else :
        plt.title(f'Image in Tensor format | Class: {label}')
    plt.show()

def dist(datasets_13, names_13, classes_13):
    for dataset_13, name_13 in zip(datasets_13, names_13):

        labels_13 = []
        for _, label_13 in dataset_13:
            labels_13.append(label_13.argmax(dim=0).item())

        print(name_13)
        print(f"\tClass: {classes_13[0]} | Count: {labels_13.count(0)}")
        print(f"\tClass: {classes_13[1]} | Count: {labels_13.count(1)}")

def score(model):
    y_pred_15 = []
    y_true_15 = []
    for image_15, label_15 in test_dataset:
        pred_15 = model.predict(image_15)
        y_pred_15.append(pred_15)
        y_true_15.append(label_15.argmax(dim=0).item())

    cm = confusion_matrix(y_true=y_pred_15, y_pred=y_true_15)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=dataset.classes)

    disp.plot(cmap='Blues')
    plt.title('AlexNet Confusion Matrix')
    plt.show()

DATA_ROOT = 'src/data/harvard/'

BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3

dataset = ImageFolder(root=DATA_ROOT, transform=ToTensor())
train_subset, test_subset = random_split(dataset, [0.80, 0.2])

transforms = [
    RandomRotation(degrees=(-15, 15)),
    Compose([RandomCrop(size=(200, 200)), Resize(size=(256,256))]),
    ElasticTransform(alpha=50.0)
]

train_dataset = data_augmentation(train_subset, transforms)

test_dataset = CustomDataset(test_subset)

train_loader = DataLoader(train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True)
test_loader = DataLoader(test_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False)

model = AlexNet()
model.fit(train_loader=train_loader, eval_loader=test_loader, epochs=EPOCHS, lr=LR, debug=True)

dist([train_dataset, test_dataset], ['Train', 'Test'], dataset.classes)

score(model)