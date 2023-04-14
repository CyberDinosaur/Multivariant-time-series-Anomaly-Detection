import hydra

from datasets import CPS_data_loader


""" Make the fundamental structures of the pipeline.
"""
def make(config, device):
    # Firstly make the dataloaders.
    dataset_cfg = config.dataset
    train_loader, val_loader, test_loader, n_sensor = CPS_data_loader(dataset_cfg)

    # Make the model
    alg_cfg = config.algorithm
    model = hydra.utils.instantiate(alg_cfg['model'], n_features=dataset_cfg['valid_features'], latent_dim=alg_cfg['latent_dim'], device=device)
    
    # Make the loss and optimizer
    train_cfg = config.train
    criterion = hydra.utils.instantiate(train_cfg['criterion'])
    optimizer = hydra.utils.instantiate(train_cfg['optimizer'], params=model.parameters(), lr=train_cfg['optimizer']['lr'])
    
    return model, train_loader, val_loader, test_loader, criterion, optimizer