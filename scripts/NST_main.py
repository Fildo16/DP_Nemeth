import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image
import time
import copy



def image_loader(image_name, imsize=112):
    
    """
    Function loads and preprocesses an image for deep learning tasks, returning a PyTorch tensor.
    """
    
    loader = transforms.Compose([transforms.Resize(imsize),
                                 transforms.CenterCrop(imsize),
                                 transforms.ToTensor()])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return loader(Image.open(image_name)).unsqueeze(0).to(device, torch.float)



class ContentLoss(nn.Module):
    
    """
    Function is a class for calculating the content loss between the input and target feature maps, 
    while gram_matrix computes the Gram matrix of the input feature map.
    """

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
    
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)  
    G = torch.mm(features, features.t())  

    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
   
    """
    Function computes the style loss between the input and target Gram matrices.
    """

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    
    """
    Function normalizes an image tensor using the mean and standard deviation
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
    


def get_input_optimizer(input_img):
    
    """
    Function initializes an optimizer for optimizing the input image.
    """
    optimizer = optim.LBFGS([input_img])
    return optimizer


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img):
    
    """
    Function constructs a neural network model and computes style and content losses for a given CNN, 
    normalization parameters, style image, and content image.
    """

    content_layers = ['conv_4']  
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']  

    normalization = Normalization(normalization_mean, normalization_std)


    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

#%% Run NST

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=1000,
                       style_weight=1000000, content_weight=1):
    
    """
    Function that runs style transfer
    Grid search with creation of images with lowest combination style and content loss
    """
    
    print('Constructing the style transfer model...')


    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    best_loss = float('inf') 
    best_output = None
    lowest_combination_iteration = None 

    style_loss_values = []  
    content_loss_values = [] 

    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            style_loss_values.append(style_score.item())  
            content_loss_values.append(content_score.item())  

            return style_score + content_score

        optimizer.step(closure)


        current_loss = style_loss_values[-1] + content_loss_values[-1]
        if current_loss < best_loss:
            best_loss = current_loss
            lowest_combination_iteration = run[0]  
            best_output = input_img.clone()

    with torch.no_grad():
        best_output.clamp_(0, 1)

    print("Iteration with lowest combination of style and content loss:", lowest_combination_iteration)

    return best_output, style_loss_values, content_loss_values





