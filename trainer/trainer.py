from trainer.base_trainer import BaseTrainer
import torch.nn as nn
from tqdm.auto import tqdm
from utils.utils import imsshow
class SimpleTrainer(BaseTrainer):
    def __init__(self, train_dl, valid_dl, vasf_model, optimizer, dev):
        super().__init__()
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.model = vasf_model
        self.criterion = nn.MSELoss()
        self.optimizer = optimizer
        self.dev = dev
        self.model = self.model.to(dev)

    def train(self, num_iter, log_every_step, dsc_len):
        for i, batch in tqdm(enumerate(self.train_dl)):
            images = batch['images'].float().to(self.dev)
            self.optimizer.zero_grad()
            vasf_result = self.model.reconstruct(images, dsc_len)
            commit_loss = vasf_result['commit_loss'].mean()
            reconstructed = vasf_result['output']
            loss = self.criterion(reconstructed, images) + 0*commit_loss
            loss.backward()
            self.optimizer.step()
            if (i+1) % log_every_step == 0:
                print(loss.item())
                self.visualization_validation()
            if (i+1) == num_iter:
                break
    

    def visualization_validation(self):
        images = next(iter(self.valid_dl))['images'].float().to(self.dev)
        imsshow(images[:4].cpu())
        vasf_result = self.model.reconstruct(images, 2)
        reconstructed = vasf_result['output']
        imsshow(reconstructed[:4].detach().cpu())

