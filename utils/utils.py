import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    """
        numbers are between -1 and 1
    """
    img = img / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()