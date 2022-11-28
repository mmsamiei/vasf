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

batch_size = 128
emb_size = 32
train_dl = RectanglesDataLoader((36,36),(0,5),(8,24), batch_size, 1)
valid_dl = RectanglesDataLoader((36,36),(0,5),(8,24), batch_size, 1)
m1 = feature_extractor.CNNFeatureExtractor(emb_size)
m2 = descriptor.NonAutoregressiveDescriptor(emb_size, emb_size, emb_size)
m3 = decoder.ImageGenerator(emb_size, emb_size)
vq = VectorQuantize(dim = emb_size, codebook_size = 512, decay = 0.995, commitment_weight = 0.25, use_cosine_sim = True, threshold_ema_dead_code = 0.01)
vasf_model = vasf.Vasf(m1, m2, m3, None).float()
loger = logger.Logger('experiments/easy22')
loger.print("number of model parameters: "+str(count_parameters(vasf_model)))
optim = torch.optim.Adam(vasf_model.parameters(), lr=1e-4)
my_trainer = manual_trainer.ManualTrainer(train_dl, valid_dl, vasf_model, optim, dev, loger)
# train_dl.set_attrs(None, (0, 1), (14,20))
my_trainer.train(500000, 5000, 3)
