from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch import Generator
from torch import device
from torch import cuda
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import SGD
from torch import max as tensor_max
from torch import sum as tensor_sum
from datetime import datetime
import tqdm

data_root = 'data/harvard'
generator = Generator().manual_seed(42)

class Modelo:

    if cuda.is_available():
        __device__ = device('cuda')
    else:
        __device__ = device('cpu')

    def __init__(self, train_loader, debug=False) -> None:        
        self.model = vgg16(weights=VGG16_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier[6] = Linear(4096,2)
        self.model.classifier[6].requires_grad = True

        self.model.to(self.__device__)

        self.train_loader = train_loader

        # Adicionar trecho de sanidade do código (plotar imagens)
        if debug:
            print('The model device is:', self.__device__)
            images, labels = next(iter(self.train_loader))
            print('The data was loaded successfully.')

    def fit(self, epochs=3, lr=0.001, debug=True):
        optimizer = SGD(self.model.parameters(),lr=lr)
        criterion = CrossEntropyLoss()
        
        now = datetime.now()
        suffix = now.strftime("%Y%m%d_%H%M%S")
        prefix = suffix # if prefix is None else prefix + '-' + suffix  

        for epoch in range(epochs):
            self.model.train()
            for idx, (train_x, train_label) in enumerate(self.train_loader):
                train_x = train_x.to(self.__device__)
                train_label = train_label.to(self.__device__)

                predict_y = self.model( train_x )
                
                # Loss:
                error = criterion( predict_y , train_label )

                # Back propagation
                optimizer.zero_grad()
                error.backward()
                optimizer.step()
                
                # Accuracy:
                predict_ys = tensor_max( predict_y, axis=1 )[1]
                correct    = tensor_sum(predict_ys == train_label)

                if debug and idx % 10 == 0 :
                    print( f'idx: {idx:4d}, _error: {error.cpu().item():5.2f}' )

        return self.model
    
__transform__ = Compose([
        ToTensor(),
    ])

# Corrigir
# __inverse_transform__ = Compose([
#     ToTensor
# ])

dataset = ImageFolder(root=data_root, transform=__transform__)

train_dataset, test_dataset = random_split(dataset, [0.3, 0.7], generator)

train_loader = DataLoader(train_dataset,
                        batch_size=4,
                        shuffle=True)
test_loader = DataLoader(test_dataset,
                        batch_size=4,
                        shuffle=False)

x = Modelo(train_loader, debug=True)
x.fit()

for test_data, test_label in test_loader:
    out = x.model(test_data)
    print(out, test_label)