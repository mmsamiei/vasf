import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    """
        numbers are between -1 and 1
    """
    img = img / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def imsshow(imgs, fig_size=(6,4)):
    num_images = imgs.shape[0]
    fig, ax = plt.subplots(1,num_images)
    fig.set_size_inches(fig_size)
    fig.set_dpi(100)
    for i, img in enumerate(imgs):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        ax[i].imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()