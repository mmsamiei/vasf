from trainer.base_trainer import BaseTrainer

class SimpleTrainer(BaseTrainer):
    def __init__(self, train_dl, valid_dl, model):
        super().__init__()
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.model = model
        
