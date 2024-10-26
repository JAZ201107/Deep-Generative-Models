import torch 
import numpy as np 
import random

from utils import get_device

device = get_device()

class Sampler:
    def __init__(self, model, img_shape, sample_size, max_len = 8192):
        self.model = model 
        self.img_shape = img_shape 
        self.sample_size = sample_size 
        self.max_len = max_len
        self.examples = [(torch.randn(1, *img_shape))*2 -1 for _ in range(sample_size)]
    
    def sample_new_exmps(self, steps = 60, step_size = 10):
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new, *self.img_shape))*2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k = self.sample_size - n_new), dim = 0)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim = 0).detach().to(device)
        
        # Perform MCMC sampling 
        inp_imgs = Sampler.generate_samples(self.model, inp_imgs, steps = steps, steps_size = step_size)
        
        # Add new images to hte buffer and remove old ones if needed 
        self.examples = list(inp_imgs.cpu().detach().chunk(self.sample_size, dim = 0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_imgs
        
    @staticmethod
    def generate_samples(model, inp_imgs, steps = 60, steps_size = 10, return_img_per_step = False):   
        is_training = model.training 
        model.eval()
        model.requires_grad_(False)
        inp_imgs.requires_grad_(True)
        
        has_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        # Use Buffer tensor in which generate noise each loop iteration.
        # More efficient than creating a new tensor each iteration.
        noise = torch.randn_like(inp_imgs).to(inp_imgs.device)
        
        imgs_per_step = []
        for _ in range(steps):
            noise.normal_(0, 0.005) # generate noise 
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(-1, 1) 
            
            out_imgs = -model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients 
            
            # Apply gradients to current samples 
            inp_imgs.data.add_(-steps_size * inp_imgs.grad.data )
            inp_imgs.grad.detach_() # Create a new computational graph
            inp_imgs.grad.zero_() # Reset the gradients
            inp_imgs.data.clamp_(-1, 1) 

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())
        
        model.requires_grad_(True)
        model.train(is_training)
        torch.set_grad_enabled(has_gradients_enabled)
        
        if return_img_per_step:
            return torch.cat(imgs_per_step) # TODO: Check if this is correct
        else:
            return inp_imgs.clone().detach()

# Why `*2 - 1`?
# 1. The input image is normalized to the range of [-1, 1].
# The Neural Network often benefit from input values that are centered around 0. 
# 3. In the Energy-Based Models, having input between[-1,1] avoid vanishing or exploding gradients problem. 