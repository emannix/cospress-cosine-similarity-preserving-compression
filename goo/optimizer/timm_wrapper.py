from timm.optim import create_optimizer
from pdb import set_trace as pb

def timm_wrapper(args, model):
    optimizer = create_optimizer(args, model)
    return optimizer

