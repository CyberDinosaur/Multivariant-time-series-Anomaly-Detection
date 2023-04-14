import os
import torch
import numpy as np
import wandb
import omegaconf

from pipeline.make import make
from pipeline.train import train
from pipeline.validate import validate

""" The overall pipeline 

    Train the model based on the hyperparameters and validate it between a certain interval.

    Args:
        hyper_params (dict): The hyperparameters to train the model.
        device (torch.device): The device to train the model on.
        seed (int): The seed to use for the random number generator.
        
    Returns:
        return the trained model.
"""
def pipeline(hyper_params, device, seed):

    # Firstly, split the parameters for sepcification
    wandb_cfg = hyper_params.wandb
    alg_cfg = hyper_params.algorithm
    dataset_cfg = hyper_params.dataset
    train_cfg = hyper_params.train
    
    # Initialize the wandb logger.
    wandb.init(
        project = wandb_cfg.project,
        name = f"{alg_cfg.alg_name} on {dataset_cfg.dataset_name} with {train_cfg.method_name}——{seed}",
        config = omegaconf.OmegaConf.to_container(hyper_params, resolve=True, throw_on_missing=True),
    )
    
    # Access all HPs through wandb.config, so that logging matches execution.
    config = wandb.config

    # Make the model, dataloader, and optimizer based on the configuration
    model, train_loader, val_loader, test_loader, criterion, optimizer = make(config, device)

    # and use them to train the model
    train_cfg = config.train
    model_type = config.algorithm['model_type'] # input_size and type
    train(model, train_loader, criterion, optimizer, train_cfg, model_type, device)

    # and test its final performance
    # validate(model, val_loader, device)
    
    
    return model