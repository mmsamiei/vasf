from data_loader.rectangles import RectanglesDataLoader
from model import feature_extractor
from model import descriptor
from model import decoder
from torch.optim import Adam
from model import vasf
import torch
from trainer import trainer
import random 
from logger import logger
from utils.utils import positionalencoding1d
from utils.utils import count_parameters

dev = torch.device('cuda')
train_dl = RectanglesDataLoader((32,32),(0,2),(4,10), 128, 1)
m1 = feature_extractor.CNNFeatureExtractor(64)
m2 = descriptor.AutoregressiveMaskedDescriptor(64, 64, 64)
m3 = decoder.ImageGenerator(64, 64)
vasf_model = vasf.Vasf(m1, m2, m3).float()
loger = logger.Logger('experiments/test')
loger.print("number of model parameters: "+str(count_parameters(vasf_model)))
optim = torch.optim.Adam(vasf_model.parameters(), lr=3e-5)
my_trainer = trainer.SimpleTrainer(train_dl, train_dl, vasf_model, optim, dev, loger)
train_dl.set_attrs(None, (0, 2), (14,16))
my_trainer.train(100, 25, 3)