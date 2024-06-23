from torch import device
from torch import cuda
from torch.nn import Module
from torchvision.models import alexnet, AlexNet_Weights
from torch.nn import Linear, BCELoss
from torch.optim import SGD
from torch.nn.functional import softmax
from torch import float
from torch import save
from tqdm import tqdm
import matplotlib.pyplot as plt

class AlexNet(Module):
    if cuda.is_available():
        _DEVICE = device('cuda')
    else:
        _DEVICE = device('cpu')

    def __init__(self) -> None:
        super().__init__()

        self.model = alexnet(weights=AlexNet_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False
       
        self.model.classifier[6] = Linear(4096,2)

        self.model.classifier[6].requires_grad = True

        self = self.to(self._DEVICE)
    
    def foward(self, x):
        return self.model(x)

    def fit(self, train_loader, test_loader=None, epochs=10, lr=1e-3, file_path='src/models/best_model.pt', debug=False):
        optimizer = SGD(self.parameters(), lr=lr)
        criterion = BCELoss()

        max_acc = 0.0
        accuracies = []
        
        for epoch in tqdm(range(epochs), desc='Training epochs'):
            self.train()
            for inputs, labels in train_loader:
                inputs = inputs.to(self._DEVICE, dtype=float)
                labels = labels.to(self._DEVICE, dtype=float)

                optimizer.zero_grad()

                # TODO: Verificar se está correto
                outputs = self.foward(inputs)
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
                    save(self.state_dict(), file_path)
                    print(f"Saving new model in epoch {epoch} with new best accuracy in test:", accuracy)

        if debug:
            plt.title('Accuracy in train')
            plt.plot(accuracies)
            plt.show()

        return self
    
    def validate(self, loader):
        self.eval()

        n_corrects = 0
        n_total = 0
        
        for inputs, labels in loader:
            inputs = inputs.to(self._DEVICE, dtype=float)
            labels = labels.to(self._DEVICE, dtype=float)

            outputs = self.foward(inputs)
            preds = softmax(outputs, dim=1).max(dim=1).indices

            n_total += len(inputs)
            n_corrects += (preds == labels).sum().item()
        
        return 100 * (n_corrects/n_total)
    
    def predict(self, image):
        self.eval()

        image = image.unsqueeze(dim=0)
        output = self.foward(image)
        preds = softmax(output, dim=1).max(dim=1).indices.item()
        
        return preds