from timm.scheduler import create_scheduler
from pdb import set_trace as pb

def timm_wrapper(args, optimizer):
    lr_scheduler, _ = create_scheduler(args, optimizer)
    return lr_scheduler

