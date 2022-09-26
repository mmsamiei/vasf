class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self):
        pass
    def _train_epoch(self, epoch):
        pass
    def train(self):
        pass
    def _save_checkpoint(self, epoch, save_best=False):
        pass
    def _resume_checkpoint(self, resume_path):
        pass