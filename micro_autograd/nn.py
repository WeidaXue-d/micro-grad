import random
from micro_autograd.engine import Value

class Neuron:

    def __init__(self,nin):        # number of input
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        out = act.tanh()

        return out
    

