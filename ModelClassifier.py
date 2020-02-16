import os
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import random
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



class ModelClassifier():
    def __init__(self):
        nn_filename = 'classifier_pizza.pth'

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        learning_rate = checkpoint['learning_rate']
        model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
        model.classifier = checkpoint['classifier']
        model.epochs = checkpoint['epochs']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)
        optimizer.load_state_dict(checkpoint['optimizer'])

        return model, optimizer

    def imshow(self, image, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots()

        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.transpose((1, 2, 0))

        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)

        ax.imshow(image)

        return ax

    def process_image(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''

        # TODO: Process a PIL image for use in a PyTorch model
        im = Image.open(image)
        im = im.resize((256, 256))
        value = 0.5 * (256 - 224)
        im = im.crop((value, value, 256 - value, 256 - value))
        im = np.array(im) / 255

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        im = (im - mean) / std

        return im.transpose(2, 0, 1)

    def predict(self, image_path, model, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''

        # TODO: Implement the code to predict the class from an image file
        # move the model to cuda
        print("AHHH")
        cuda = torch.cuda.is_available()
        print(cuda)
        if cuda:
            # Move model parameters to the GPU
            model.cuda()
            print("Number of GPUs:", torch.cuda.device_count())
            print("Device name:", torch.cuda.get_device_name(torch.cuda.device_count() - 1))
        else:
            model.cpu()
            print("We go for CPU")

        # turn off dropout
        model.eval()
        print("ah")
        # The image
        image = self.process_image(image_path)
        print(image)
        # tranfer to tensor
        image = torch.from_numpy(np.array([image])).float()

        # The image becomes the input
        image = Variable(image)
        if cuda:
            image = image.cuda()

        output = model.forward(image)

        probabilities = torch.exp(output).data

        # getting the topk (=5) probabilites and indexes
        # 0 -> probabilities
        # 1 -> index
        prob = torch.topk(probabilities, topk)[0].tolist()[0]  # probabilities
        index = torch.topk(probabilities, topk)[1].tolist()[0]  # index
        print("index : ::", index)

        ind = []
        for i in range(len(model.class_to_idx.items())):
            ind.append(list(model.class_to_idx.items())[i][0])

        # transfer index to label
        label = []
        for i in range(2):
            label.append(ind[index[i]])
        return prob, label