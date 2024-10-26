from network import CNNModel
from sampler import Sampler



class Model:
    def __init__(self, img_shape, batch_size, alpha = 0.1, lr = 1e-4, beta1 = 0.0, **CNN_args):
        