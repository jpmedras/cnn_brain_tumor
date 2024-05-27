#!/usr/bin/env python
# coding: utf-8

import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch
from torch.utils.tensorboard import SummaryWriter

import torch.optim 
import matplotlib.pyplot as plt
  
from datetime import datetime

from tqdm import tqdm

import copy

tensorboard_path = 'board/'
models_path = 'saves/'

def my_tensor_image_show ( image , label=None ):
    image = image.numpy().transpose((1, 2, 0))
    plt.imshow(image)
    plt.axis('off')
    if label is None :
        plt.title('Image in tensor format.')
    else :
        plt.title(f'Image in tensor format | Class: {label:2d}')
    plt.show()

data_root = 'data/harvard'
generator = torch.Generator().manual_seed(42)

__transform__ = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

dataset = torchvision.datasets.ImageFolder(root=data_root, transform=__transform__)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.3, 0.7], generator)

batch_size = 32

train_tensors = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
test_tensors = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

images, labels = next(iter(train_tensors))
my_tensor_image_show(images[0], label=labels[0])

images, labels = next(iter(test_tensors))
my_tensor_image_show(images[0], label=labels[0])

def plot_layers ( net , writer, epoch ) :
    layers = list(net.classifier.modules())
    
    layer_id = 1
    for layer in layers:
        if isinstance(layer, torch.nn.Linear) :

            writer.add_histogram('Bias/conv{}'.format(layer_id), layer.bias, 
                                epoch )
            writer.add_histogram('Weight/conv{}'.format(layer_id), layer.weight, 
                                epoch )
            writer.add_histogram('Grad/conv{}'.format(layer_id), layer.weight.grad, 
                                    epoch )
            layer_id += 1


def train ( train_loader, test_loader, net, dataset_size, my_device='cpu',
           prefix=None, upper_bound=100.0, save=False, epochs=100, 
           lr=1e-1, device='cpu', debug=False, layers2tensorboard=False , batch_size=64) :

    optimizer = torch.optim.SGD(net.parameters(),lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    now = datetime.now()
    suffix = now.strftime("%Y%m%d_%H%M%S")
    prefix = suffix if prefix is None else prefix + '-' + suffix  

    # writer = SummaryWriter( log_dir=tensorboard_path+prefix )
        
    accuracies = []
    max_accuracy = -1.0  

    for epoch in tqdm(range(epochs), desc='Training epochs...') :
        net.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)

            predict_y = net( train_x )
            
            # Loss:
            error = criterion( predict_y , train_label )

            # writer.add_scalar( 'Loss/train', error.cpu().item(), 
            #                     idx+( epoch*(dataset_size//batch_size) ) )

            # Back propagation
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
            
            # Accuracy:
            predict_ys = torch.max( predict_y, axis=1 )[1]
            correct    = torch.sum(predict_ys == train_label)

            # writer.add_scalar( 'Accuracy/train', correct/train_x.size(0), 
            #                     idx+( epoch*(dataset_size//batch_size) ) )

            if debug and idx % 10 == 0 :
                print( f'idx: {idx:4d}, _error: {error.cpu().item():5.2f}' )

        # if layers2tensorboard :
        #     plot_layers( net, writer, epoch )

        accuracy = validate(net, test_loader, device=device)
        accuracies.append(accuracy)
        # writer.add_scalar( 'Accuracy/test', accuracy, epoch )
        
        if accuracy > max_accuracy :
            best_model = copy.deepcopy(net)
            max_accuracy = accuracy
            print("Saving Best Model with Accuracy: ", accuracy)
        
        print( f'Epoch: {epoch+1:3d} | Accuracy : {accuracy:7.4f}%' )

        if accuracy > upper_bound :
            break
    
    if save : 
        path = f'{models_path}AlexNet-{dataset}-{max_accuracy:.2f}.pkl'
        torch.save(best_model, path)
        print('Model saved in:',path)
    
    plt.plot(accuracies)

    # writer.flush()
    # writer.close()
    
    return best_model

def validate ( model , data , device='cpu') :

    model.eval()

    correct = 0
    sum = 0
    
    for idx, (test_x, test_label) in enumerate(data) : 
        test_x = test_x.to(device)
        test_label = test_label.to(device)
        predict_y = model( test_x ).detach()
        predict_ys = torch.max( predict_y, axis=1 )[1]
        sum = sum + test_x.size(0)
        correct = correct + torch.sum(predict_ys == test_label)
        correct = correct.cpu().item()
    
    return correct*100./sum

model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
model.eval()

for param in model.parameters():
    param.requires_grad = False

model.classifier[6] = torch.nn.Linear(4096,2)
model.classifier[6].requires_grad = True
# model.classifier[4].requires_grad = True
# model.classifier[1].requires_grad = True

if torch.cuda.is_available():
    my_device = torch.device("cuda:0")
    print("Running on CUDA.")
else:
    my_device = torch.device("cpu")
    print("No Cuda Available")
    
my_device = 'cpu'
model = model.to(my_device)

epochs = 100
lr = 1e-3
dataset = 'figshare'
prefix = 'AlexNet-TL-{}-e-{}-lr-{}'.format(dataset, epochs, lr)

torch.cuda.empty_cache()

net = train(train_tensors, test_tensors, model, len(train_dataset),
            epochs=epochs, device=my_device, save=True, 
            prefix=prefix, lr=lr, layers2tensorboard=True, batch_size=batch_size)

