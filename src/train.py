import numpy as np
from torch import manual_seed
import random
from torchvision.transforms import Compose, ToTensor, ColorJitter, GaussianBlur, Normalize
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from torch import stack
import matplotlib.pyplot as plt
from alexnet_pretrained import AlexNet

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def plot_img(image, label=None):
    plt.imshow(image.permute((1, 2, 0)), cmap='gray')
    plt.axis('off')
    if label is None :
        plt.title('Image in Tensor format')
    else :
        plt.title(f'Image in Tensor format | Class: {label}')
    plt.show()

np.random.seed(42)
manual_seed(42)
random.seed(42)

data_root = 'src/data/harvard'

dataset_mean = (0.1338, 0.1338, 0.1338)
dataset_std = (0.2054, 0.2054, 0.2054)

dataset = ImageFolder(root=data_root, transform=Compose([ToTensor()]))

train_subset, test_subset = random_split(dataset, [0.6, 0.4])

train_dataset = DatasetFromSubset(
    subset=train_subset,
    transform=Normalize(mean=dataset_mean, std=dataset_std)
)

test_dataset = DatasetFromSubset(
    subset=test_subset,
    transform=Normalize(mean=dataset_mean, std=dataset_std)
)

jitter_subset = Subset(dataset, train_subset.indices)
train_jitter = DatasetFromSubset(
    subset=jitter_subset,
    transform=Compose([ColorJitter(brightness=.3, hue=.3), Normalize(mean=dataset_mean, std=dataset_std)])
)

train_dataset = ConcatDataset([train_dataset, train_jitter])

for s_dataset in [train_dataset, test_dataset]:
    labels = [label for image, label in s_dataset]

    print(f"Class: {dataset.classes[0]} | Count: {labels.count(0)}")
    print(f"Class: {dataset.classes[1]} | Count: {labels.count(1)}")

train_loader = DataLoader(train_dataset,
                        batch_size=32,
                        shuffle=True)
test_loader = DataLoader(test_dataset,
                        batch_size=32,
                        shuffle=False)

model = AlexNet()
model.fit(train_loader=train_loader, test_loader=test_loader, epochs=15, lr=1e-3, debug=True)