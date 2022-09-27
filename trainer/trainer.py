from trainer.base_trainer import BaseTrainer
import torch.nn as nn
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

    def train(self, num_iter):
        for i, batch in enumerate(self.train_dl):
            images = batch['images'].float().to(self.dev)
            self.optimizer.zero_grad()
            vasf_result = self.model.reconstruct(images, 2)
            commit_loss = vasf_result['commit_loss'].mean()
            reconstructed = vasf_result['output']
            loss = self.criterion(reconstructed, images) + commit_loss
            loss.backward()
            self.optimizer.step()
            if i == num_iter:
                break
