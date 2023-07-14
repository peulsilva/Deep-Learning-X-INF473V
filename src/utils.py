import torch
import numpy as np

def get_num_parameters(model : torch.nn.Module) -> int:
    """Given a neural network, calculate number of trainable params

    Args:
        model (torch.nn.Module): neural network

    Returns:
        int: number of params 
    """    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    return params