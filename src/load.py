from alexnet_pretrained import AlexNet
from torch import load
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

model = AlexNet()
model.load_state_dict(load('src/weights/best_model.pt'))

dataset = ImageFolder(root='src/data/', transform=ToTensor())

for idx, (image, label) in enumerate(dataset):
    if idx % 10 == 0:
        pred = model.predict(image)
        print(f"Prediction: {pred} | Label: {label}")