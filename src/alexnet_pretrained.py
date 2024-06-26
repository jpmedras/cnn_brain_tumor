from torch import device
from torch import cuda
from torch.nn import Module
from torchvision.models import alexnet, AlexNet_Weights
from torch.nn import Linear, BCEWithLogitsLoss
from torch.optim import SGD
from torch.nn.functional import softmax
from torch import no_grad
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

    def fit(self, train_loader, eval_loader=None, epochs=10, lr=1e-3, file_path='src/weights/best_model.pt', debug=False):
        optimizer = SGD(self.parameters(), lr=lr)
        criterion = BCEWithLogitsLoss()

        max_acc = 0.0
        train_accuracies = []
        test_accuracies = []
        
        for epoch in tqdm(range(epochs), desc='Training epochs'):
            self.train()

            n_corrects = 0
            n_total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(self._DEVICE, dtype=float)
                labels = labels.to(self._DEVICE, dtype=float)

                optimizer.zero_grad()

                outputs = self.foward(inputs)
                # print(outputs.shape)
                # print(labels.shape)
                loss = criterion(outputs, labels)

                y_pred = softmax(outputs, dim=1).max(dim=1).indices
                y_true = labels.argmax(dim=1)
                # print(y_pred)
                # print(y_true)
                # print()
                n_total += len(inputs)
                n_corrects += (y_pred == y_true).sum().item()

                # backward
                loss.backward()
                optimizer.step()

            train_accuracies.append(100 * (n_corrects/n_total))
            
            if eval_loader is not None:
                accuracy = self.validate(eval_loader)

                test_accuracies.append(accuracy)

                if accuracy > max_acc:
                    max_acc = accuracy
                    save(self.state_dict(), file_path)
                    print(f"Saving new model in epoch {epoch} with new best accuracy in test:", accuracy)

        if debug:
            plt.figure()
            plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
            plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('AlexNet accuracies')
            plt.savefig(f'alexnet_accuracies.png')
            plt.show()

        return self
    
    def validate(self, loader):
        self.eval()

        n_corrects_123 = 0
        n_total_123 = 0

        with no_grad():

            for inputs_123, labels_123 in loader:
                inputs_123 = inputs_123.to(self._DEVICE, dtype=float)
                labels_123 = labels_123.to(self._DEVICE, dtype=float)


                outputs_123 = self.foward(inputs_123)
                y_pred_123 = softmax(outputs_123, dim=1).max(dim=1).indices
                y_true_123 = labels_123.argmax(dim=1)

                n_total_123 += len(inputs_123)
                n_corrects_123 += (y_pred_123 == y_true_123).sum().item()
            
            return 100 * (n_corrects_123/n_total_123)
    
    def predict(self, image_4):
        self.eval()

        image_4 = image_4.unsqueeze(dim=0)
        output_4 = self.foward(image_4)
        preds_4 = softmax(output_4, dim=1).max(dim=1).indices.item()

        return preds_4