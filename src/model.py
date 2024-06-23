import numpy as np
from torch import manual_seed
import random
from torchvision.transforms import ToTensor
from torch import device
from torch import cuda
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset, DataLoader
from torch.nn import Module
from torchvision.models import alexnet, AlexNet_Weights
from torch.nn import Conv2d, Linear, BCELoss
from torch.optim import SGD
from torch.nn.functional import softmax
from torch import float
from torch import save
from tqdm import tqdm
import matplotlib.pyplot as plt

data_root = 'src/data/harvard'

np.random.seed(42)
manual_seed(42)
random.seed(42)

def plot_img(image, label=None):
    plt.imshow(image.permute((1, 2, 0)), cmap='gray')
    plt.axis('off')
    if label is None :
        plt.title('Image in Tensor format')
    else :
        plt.title(f'Image in Tensor format | Class: {label:2d}')
    plt.show()   

class AlexNet(Module):
    if cuda.is_available():
        _DEVICE = device('cuda')
    else:
        _DEVICE = device('cpu')

    def __init__(self, debug=False) -> None:
        super().__init__()

        self.model = alexnet(weights=AlexNet_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False
       
        self.model.classifier[6] = Linear(4096,2)

        self.model.classifier[6].requires_grad = True

        if debug:
            for name, module in self.model.named_modules():
                print(name, module)

        self.model.to(self._DEVICE)

    def foward(self, x):
        return self.model(x)

    def fit(self, train_loader, test_loader=None, epochs=10, lr=1e-3, file_path='src/best_model.pt', debug=False):
        optimizer = SGD(self.model.parameters(), lr=lr)
        criterion = BCELoss()

        max_acc = 0.0
        accuracies = []
        
        for epoch in tqdm(range(epochs), desc='Training epochs'):
            self.model.train()
            for inputs, labels in train_loader:
                inputs = inputs.to(self._DEVICE, dtype=float)
                labels = labels.to(self._DEVICE, dtype=float)

                optimizer.zero_grad()

                # TODO: Verificar se está correto
                outputs = self.model(inputs)
                outputs = softmax(outputs, dim=1).max(dim=1).values
                loss = criterion(outputs, labels)

                # backward
                loss.backward()
                optimizer.step()
            
            if test_loader is not None:
                accuracy = self.validate(test_loader)

                accuracies.append(accuracy)

                if accuracy > max_acc:
                    max_acc = accuracy
                    save(self.model.state_dict(), file_path)
                    print(f"Saving new model in epoch {epoch} with new best accuracy in test:", accuracy)

        if debug:
            plt.title('Accuracy in train')
            plt.plot(accuracies)
            plt.show()

        return self.model
    
    def validate(self, loader):
        self.model.eval()

        corrects = 0
        total_len = 0
        
        for inputs, labels in loader:
            inputs = inputs.to(self._DEVICE, dtype=float)
            labels = labels.to(self._DEVICE, dtype=float)

            outputs = self.model(inputs)
            preds = softmax(outputs, dim=1).max(dim=1).indices

            total_len += len(inputs)
            corrects += (preds == labels).sum().item()
        
        return 100 * (corrects/total_len)

dataset = ImageFolder(root=data_root, transform=ToTensor())

train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset,
                        batch_size=32,
                        shuffle=True)
test_loader = DataLoader(test_dataset,
                        batch_size=32,
                        shuffle=False)

model = AlexNet(debug=False)
model.fit(train_loader=train_loader, test_loader=test_loader, epochs=35, lr=1e-3, debug=True)