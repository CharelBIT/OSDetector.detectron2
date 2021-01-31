from detectron2.engine import DefaultTrainer

class YOLOTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
