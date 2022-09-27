from trainer.base_trainer import BaseTrainer
import torch.nn as nn
class SimpleTrainer(BaseTrainer):
    def __init__(self, train_dl, valid_dl, vasf_model, optimizer):
        super().__init__()
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.model = vasf_model
        self.criterion = nn.MSELoss()
        self.optimizer = optimizer

    def train(self, num_iter):
        for i, batch in enumerate(self.train_dl):
            images = batch['images'].float()
            self.optimizer.zero_grad()
            reconstructed = self.model.reconstruct(images, 2)['output']
            loss = self.criterion(reconstructed, images)
            loss.backward()
            self.optimizer.step()
            if i == num_iter:
                break
