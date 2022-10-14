from trainer.base_trainer import BaseTrainer
import torch.nn as nn
from tqdm.auto import tqdm
from utils.utils import imsshow
import numpy as np
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
        self.model.train()
        loss_history = []
        for i, batch in tqdm(enumerate(self.train_dl)):
            images = batch['images'].float().to(self.dev)
            self.optimizer.zero_grad()
            vasf_result = self.model.reconstruct(images, dsc_len)
            commit_loss = vasf_result['commit_loss'].mean()
            reconstructed = vasf_result['output']
            loss = self.criterion(reconstructed, images) + 0*commit_loss
            loss.backward()
            self.optimizer.step()
            loss_history.append(loss.item())
            if (i+1) % log_every_step == 0:
                print(f"iteration {i+1} avg loss: {np.mean(loss_history).round(4)}")
                loss_history = []
                self.visualization_validation()
            if (i+1) == num_iter:
                break
    

    def visualization_validation(self):
        self.model.eval()
        images = next(iter(self.valid_dl))['images'].float().to(self.dev)
        print("correct photo:")
        imsshow(images[:4].cpu(), fig_size=(4.5,3))
        vasf_result = self.model.reconstruct(images, 1)
        reconstructed = vasf_result['output']
        mse = np.around(self.criterion(reconstructed, images).mean().item(), 4)
        print(f"reconstructed by one token, mse: {mse}")
        imsshow(reconstructed[:4].detach().cpu(), fig_size=(4.5,3))
        vasf_result = self.model.reconstruct(images, 2)
        reconstructed = vasf_result['output']
        mse = np.around(self.criterion(reconstructed, images).mean().item(), 4)
        print(f"reconstructed by two tokens, mse: {mse}")
        imsshow(reconstructed[:4].detach().cpu(), fig_size=(4.5,3))
        vasf_result = self.model.reconstruct(images, 3)
        reconstructed = vasf_result['output']
        mse = np.around(self.criterion(reconstructed, images).mean().item(), 4)
        print(f"reconstructed by three tokens, , mse: {mse}")
        imsshow(reconstructed[:4].detach().cpu(), fig_size=(4.5,3))

