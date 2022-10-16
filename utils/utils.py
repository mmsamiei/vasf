import matplotlib.pyplot as plt
import numpy as np
import torch
import math 
from matplotlib import gridspec


def imshow(img, fig_size=(6,4)):
    """
        numbers are between -1 and 1
        [c, h, w]
    """
    img = img / 2 + 0.5     # unnormalize
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size)
    ax.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
    return fig


def imsshow(img, fig_size=(6,4)):
    '''
        give numpy array
    '''
    num_dim = len(img.shape)
    if num_dim == 3:
        return imshow(img, fig_size)
    elif num_dim == 4:
        return imsshow_4d(img, fig_size)
    elif num_dim == 5:
        return imsshow_5d(img, fig_size)

def imsshow_4d(imgs, fig_size=(6,4)):
    '''
        [num_img, c, h, w]
    '''
    num_images = imgs.shape[0]
    fig, ax = plt.subplots(1,num_images)
    fig.set_size_inches(fig_size)
    fig.set_dpi(100)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    for i, img in enumerate(imgs):
        img = img / 2 + 0.5     # unnormalize
        #npimg = img.numpy()
        ax[i].imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
    return fig


def imsshow_5d(imgs, fig_size=(4,6)):
    '''
        [num_rows, num_img, c, h, w]
    '''
    num_rows, num_img, c, h, w = imgs.shape
    nrow, ncol = num_rows, num_img
    fig = plt.figure(figsize=(2*(ncol+1), 2*(nrow+1))) 
    gs = gridspec.GridSpec(nrow, ncol,
         wspace=0.2, hspace=0.2, 
         top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
         left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 

    fig.set_dpi(100)
    for i in range(num_rows):
        for j in range(num_img):
            img = imgs[i, j]
            img = img / 2 + 0.5     # unnormalize
            #npimg = img.numpy()
            ax = plt.subplot(gs[i,j])
            ax.imshow(np.transpose(img, (1, 2, 0)))
            ax.set_axis_off()
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    plt.show()
    return fig


def _imsshow_5d(imgs, fig_size=(6,4)):
    '''
        [num_rows, num_img, c, h, w]
    '''
    num_rows, num_img, c, h, w = imgs.shape
    fig, ax = plt.subplots(num_rows, num_img)
    fig.set_size_inches(fig_size)
    fig.set_dpi(100)
    for i in range(num_rows):
        for j in range(num_img):
            img = imgs[i, j]
            img = img / 2 + 0.5     # unnormalize
            #npimg = img.numpy()
            ax[i][j].imshow(np.transpose(img, (1, 2, 0)))
            ax[i][j].set_axis_off()
    plt.show()
    return fig


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)