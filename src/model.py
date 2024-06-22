from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, Grayscale, Normalize
from torch import Generator
from torch import device
from torch import cuda
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.models import alexnet, AlexNet_Weights
from torch.nn import Linear, BCELoss
from torch.optim import SGD
from torch.nn.functional import softmax
from torch import float
from torch import save
from tqdm import tqdm
import matplotlib.pyplot as plt

data_root = 'src/data/harvard'
generator = Generator().manual_seed(42)

def plot_img(image, label=None):
    plt.imshow(image.permute((1, 2, 0)), cmap='gray')
    plt.axis('off')
    if label is None :
        plt.title('Image in Tensor format')
    else :
        plt.title(f'Image in Tensor format | Class: {label:2d}')
    plt.show()   

class BTumor:
    if cuda.is_available():
        _DEVICE = device('cuda')
    else:
        _DEVICE = device('cpu')

    def __init__(self, debug=False) -> None:        
        self.model = alexnet(weights=AlexNet_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        if debug:
            for name, module in self.model.named_modules():
                print(name, module)

        self.model.classifier[6] = Linear(4096,2)

        self.model.classifier[6].requires_grad = True

        self.model.to(self._DEVICE)

    def fit(self, train_loader, test_loader=None, epochs=10, lr=1e-3, debug=False):
        optimizer = SGD(self.model.parameters(), lr=lr)
        criterion = BCELoss()

        max_acc = 0.0
        
        for epoch in tqdm(range(epochs), desc='Training epochs...'):
            self.model.train()
            for idx, (inputs, labels) in enumerate(train_loader):
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
                accuracy = self.validate(test_loader, debug=debug)

                if accuracy > max_acc:
                    max_acc = accuracy
                    save(self.model.state_dict(), 'src/best_model_params.pt')
                    print("Saving new model with new best accuracy in test:", accuracy)

        return self.model
    
    def validate(self, loader, debug=False):
        self.model.eval()

        corrects = 0
        total_len = 0
        
        for idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(self._DEVICE, dtype=float)
            labels = labels.to(self._DEVICE, dtype=float)

            outputs = self.model(inputs)
            preds = softmax(outputs, dim=1).max(dim=1).indices

            total_len += len(inputs)
            corrects += (preds == labels).sum().item()
        
        return 100 * (corrects/total_len)
    
_transform = Compose([
    ToTensor(),
    Normalize(
        mean=(0.1336, 0.1336, 0.1336),
        std=(0.2156, 0.2156, 0.2156)
    )
]) 

dataset = ImageFolder(root=data_root, transform=_transform)

train_dataset, test_dataset = random_split(dataset, [0.3, 0.7], generator)

train_loader = DataLoader(train_dataset,
                        batch_size=8,
                        shuffle=True,
                        generator=generator)
test_loader = DataLoader(test_dataset,
                        batch_size=8,
                        shuffle=False,
                        generator=generator)

model = BTumor(debug=False)
model.fit(train_loader=train_loader, test_loader=test_loader, epochs=20, lr=1e-1, debug=True)