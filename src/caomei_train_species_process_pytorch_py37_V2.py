from shufflenetv2 import *
from leaf_process_utils import *
from torch import optim, nn, cuda
from torchvision import datasets, transforms
from torch.utils import data
import torch
from pathlib import Path
import logging
import time
import copy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

img_size = 224
species = 'caomei'
train_folder = './train-class-desease/caomei'
valid_folder = './valid-class-desease/caomei-v'
t = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
log_path = f'./logs/{species}{t}.log'
fh = logging.FileHandler(log_path, mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.debug(f'{species}')



class TransformLeaf(object):
    def __init__(self, size=(img_size, img_size)):
        self.size = size
        
    def __call__(self, img):
        """
        Args:
            img: PIL Image
            
        Returns:
            np.array
        """
        return process_all(img, size=(img_size, img_size))


def train_model(model: nn.Module, maxepoch: int, save_name: str=None):
    # model
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print('multiple gpus used')
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # optim
    criterion = nn.CrossEntropyLoss()
    logger.debug('criterion: CrossEntropyLoss')
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    logger.debug('optimizer: RMSprop')
    
    # data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            TransformLeaf(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            TransformLeaf(),
            transforms.ToTensor(),
        ]),
    }
    
    image_datasets = {'train': datasets.ImageFolder(train_folder, data_transforms['train']),
                      'val': datasets.ImageFolder(valid_folder, data_transforms['val'])}
    dataloaders = {'train': data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=5),
                   'val': data.DataLoader(image_datasets['val'], batch_size=100, shuffle=True, num_workers=3)}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    # train and valid
    for epoch in range(maxepoch):
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                ac = torch.sum(preds == labels.data).double() / labels.data.shape[0]
                logger.debug(f'{epoch} {phase} loss: {loss.item()} acc: {ac}')
                print(f'{epoch} {phase} loss: {loss.item()} acc: {ac}')
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            logger.debug(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
      
    if save_name:
        torch.save(best_model_wts, f'./models/{species}_shufflenetv2_{save_name}_params.pkl')
        logger.info(f'./models/{species}_shufflenetv2{save_name}_params.pkl')
        
model = ShuffleNetV2(scale=1.5, in_channels=12, c_tag=0.5, num_classes=3, activation=nn.ReLU, SE=False, residual=False)

train_model(model, 80, t)

