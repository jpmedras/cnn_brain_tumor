import numpy as np
from torch import manual_seed
import random
from torchvision.transforms import Compose, ToTensor, RandomRotation, RandomCrop, Resize, ElasticTransform, ColorJitter
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch import load
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
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)
    
def data_augmentation(subset, transforms):
    datasets = [subset]

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

def dist(datasets, names, classes):
    for dataset, name in zip(datasets, names):
        labels = [label for _, label in dataset]

        print(name)
        print(f"\tClass: {classes[0]} | Count: {labels.count(0)}")
        print(f"\tClass: {classes[1]} | Count: {labels.count(1)}")

def score(model):
    y_pred = []
    y_true = []
    for image, label in test_dataset:
        pred = model.predict(image)
        y_pred.append(pred)
        y_true.append(label)

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=dataset.classes)

    disp.plot(cmap='Blues')
    plt.title('AlexNet Confusion Matrix')
    plt.show()

DATA_ROOT = 'src/data/harvard/'

BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3

dataset = ImageFolder(root=DATA_ROOT, transform=ToTensor())
train_subset, test_subset = random_split(dataset, [0.6, 0.4])

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

score(model)

dist([train_dataset, test_dataset], ['Train', 'Test'], dataset.classes)