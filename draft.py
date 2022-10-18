from data_loader.rectangles import RectanglesDataLoader
from model import feature_extractor
from model import descriptor
from model import decoder
from torch.optim import Adam
from model import vasf
import torch
from trainer import manual_trainer
import random 
from logger import logger
from utils.utils import positionalencoding1d
from utils.utils import count_parameters
import numpy as np
from vector_quantize_pytorch import VectorQuantize

dev = torch.device('cuda')
train_dl = RectanglesDataLoader((32,32),(0,2),(4,10), 128, 1)
m1 = feature_extractor.CNNFeatureExtractor(64)
m2 = descriptor.NonAutoregressiveDescriptor(64, 64, 64)
m3 = decoder.ImageGenerator(64, 64)
vq = VectorQuantize(dim = 64, codebook_size = 1024, decay = 0.99, commitment_weight = 0.1, use_cosine_sim = True, threshold_ema_dead_code = 0.01)
vasf_model = vasf.Vasf(m1, m2, m3, vq).float()
loger = logger.Logger('experiments/easy6')
loger.print("number of model parameters: "+str(count_parameters(vasf_model)))
optim = torch.optim.Adam(vasf_model.parameters(), lr=4e-5)
my_trainer = manual_trainer.ManualTrainer(train_dl, train_dl, vasf_model, optim, dev, loger)
train_dl.set_attrs(None, (0, 4), (12,20))
my_trainer.train(200000, 2000, 1)
# a = torch.rand((2,4,1,32,32)).numpy()
# from utils.utils import imsshow
# a[0,:,:,:,:] = 1
# a[0,:,:,:,0] = 1
# imsshow(a, mode='01').savefig('1.png')
