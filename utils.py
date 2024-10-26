import torch 
import numpy as np 
import random 

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def seed_everything(seed = 42):
    np.random.seed(seed)
    random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


class GenerateCallback:
    def  __init__(self, batch_size = 8, vis_steps = 8, num_steps = 256, every_n_epochs = 5):
        pass
    
    def on_epoch_end(self,    trainer):
        pass 

    def generate_imgs(self):
        pass 