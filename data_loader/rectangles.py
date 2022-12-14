from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from utils import utils
import random
import torchvision.transforms as transforms

class RectanglesDataset(IterableDataset):
    def __init__(self, image_shape, interval_num_obj, interval_rect_size):
        assert type(image_shape) == tuple
        assert type(interval_num_obj) == tuple
        assert type(interval_rect_size) == tuple
        self.image_shape = (3, image_shape[0], image_shape[1])
        self.interval_num_obj = interval_num_obj
        self.interval_rect_size = interval_rect_size
        super().__init__()

    def generate_observation(self):
        image = np.random.rand(*self.image_shape)
        image[0] = random.uniform(-1, 1)
        image[1] = random.uniform(-1, 1)
        image[2] = random.uniform(-1, 1)
        num_obj = random.randint(*self.interval_num_obj)
        for i in range(num_obj):
            rect_size = random.randint(*self.interval_rect_size)
            x, y = random.randint(0,self.image_shape[1]-rect_size), random.randint(0,self.image_shape[2]-rect_size)
            image[0, x:x+rect_size, y:y+rect_size] = random.uniform(-1, 1)
            image[1, x:x+rect_size, y:y+rect_size] = random.uniform(-1, 1)
            image[2, x:x+rect_size, y:y+rect_size] = random.uniform(-1, 1)
        return {'images': image}
    
    def set_attrs(self, image_shape=None, interval_num_obj=None, interval_rect_size=None):
        if image_shape is not None:
            self.image_shape = (3, image_shape[0], image_shape[1])
        if interval_num_obj is not None:
            self.interval_num_obj = interval_num_obj
        if interval_rect_size is not None:
            self.interval_rect_size = interval_rect_size

    def __iter__(self):
        while(True):
            yield self.generate_observation()

class RectanglesDataLoader(DataLoader):

    def __init__(self, image_shape, interval_num_obj, interval_rect_size, batch_size, num_workers=1):
        self.dataset = RectanglesDataset(image_shape, interval_num_obj, interval_rect_size)
        super().__init__(self.dataset, batch_size=batch_size, num_workers=num_workers)

        ##TODO TRANSFORM ON DATALOADER
    
    def set_attrs(self, image_shape=None, interval_num_obj=None, interval_rect_size=None):
        self.dataset.set_attrs(image_shape, interval_num_obj, interval_rect_size)