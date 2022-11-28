from codecs import iterdecode
from trainer.base_trainer import BaseTrainer
import torch.nn as nn
from tqdm.auto import tqdm
from utils.utils import imsshow
import numpy as np
from einops import rearrange
import time

class ManualTrainer(BaseTrainer):
    def __init__(self, train_dl, valid_dl, vasf_model, optimizer, dev, loger):
        super().__init__()
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.model = vasf_model
        self.criterion = nn.MSELoss()
        self.optimizer = optimizer
        self.dev = dev
        self.model = self.model.to(dev)
        self.loger = loger

    def train(self, num_iter, log_every_step, dsc_len):
        loger = self.loger
        self.model.train()
        loss_history = []
        for i, batch in tqdm(enumerate(self.train_dl), total=num_iter):
            images = batch['images'].float().to(self.dev)
            self.optimizer.zero_grad()
            quantization_weight = self.get_scheduled_quantization_weight(i)
            dsc_len = int(self.get_scheduled_desc_len(i))
            vasf_result = self.model.reconstruct(images, dsc_len, quantization_weight)
            reconstructed = vasf_result['output']
            loss = self.criterion(reconstructed, images)
            if vasf_result['vq_loss'] is not None:
                loss += vasf_result['vq_loss'].mean()
            loss.backward()
            self.optimizer.step()
            loss_history.append(loss.item())
            if (i+1) % log_every_step == 0:
                loger.print(f"iteration {i+1} avg loss: {np.mean(loss_history).round(4)}")
                loss_history = []
                self.visualization_validation(i+1, quantization_weight)
            if (i+1) == num_iter:
                break
    

    def get_scheduled_desc_len(self, iter_num):
        if iter_num < 100000:
            return 1
        elif iter_num < 200000:
            return (iter_num % 2)+1
        elif iter_num < 300000:
            return (iter_num % 3)+1
        else:
            return (iter_num % 3)+1 

    def get_scheduled_quantization_weight(self, iter_num):
        warm_start_step = 1000 
        end_step = 120000
        if iter_num < warm_start_step:
            return 0.0
        elif iter_num > end_step:
            return 1.0
        else:
            return (iter_num - warm_start_step) / (end_step-warm_start_step)
    
    def visualization_validation(self, iter_num, quantization_weight=1.0):
        loger = self.loger
        vasf_result = [None]*5
        self.model.eval()
        images = next(iter(self.valid_dl))['images'].float().to(self.dev)[:4]
        for i in range(1,5):
            vasf_result[i] = self.model.reconstruct(images, i, quantization_weight)
        row_list = [images]
        mses = []
        for i in range(1,5):
            row_list.append(vasf_result[i]['output'])
            mses.append(np.around(self.criterion(vasf_result[i]['output'], images).mean().item(), 4))
        graphs = rearrange(row_list, 'r b c h w ->r b c h w')
        figure = imsshow(graphs.detach().cpu().numpy(), fig_size=(4.5,3))
        figure.suptitle(', '.join(str(x) for x in mses), fontsize=32)
        loger.plot(figure, f'{iter_num}_a.png')
        row_list = []    
        for i in range(1,5):
            for j in range(i):
                row_list.append(vasf_result[i]['token_outputs'][:,j])
        graphs = rearrange(row_list, 'r b c h w ->r b c h w')
        figure = graphs.detach().cpu().numpy()
        figure = imsshow(graphs.detach().cpu().numpy(), fig_size=(4.5,3))
        loger.plot(figure, f'{iter_num}_b.png')

        row_list = []    
        for i in range(1,5):
            for j in range(i):
                row_list.append(vasf_result[i]['masks'][:,j])
        graphs = rearrange(row_list, 'r b c h w ->r b c h w')
        figure = imsshow(graphs.detach().cpu().numpy(), fig_size=(4.5,3), mode='01')
        loger.plot(figure, f'{iter_num}_c.png')

        del figure
        del graphs

    def _visualization_validation(self):
        loger = self.loger
        self.model.eval()
        images = next(iter(self.valid_dl))['images'].float().to(self.dev)
        print("correct photo:")
        loger.plot(imsshow(images[:4].cpu(), fig_size=(4.5,3)), 'a.png')
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

