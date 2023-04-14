import wandb
import torch


def train(model, train_loader, criterion, optimizer, train_cfg, model_type, device):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more...
    wandb.watch(model, criterion, log="all", log_freq=10)


    total_batches = len(train_loader) * train_cfg['epochs']
    example_ct = 0  # number of examples seen
    batch_ct = 0
    
    for epoch in range(train_cfg['epochs']):
        for _, (features, labels) in enumerate(train_loader):  # features: N * L * K * D
            # Average loss within a batch
            loss = train_batch(features, labels, model, optimizer, criterion, model_type, device)
            example_ct += len(features)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

def train_batch(features, labels, model, optimizer, criterion, model_type, device):
    features, labels = features.to(device), labels.to(device)
    
    # Forward pass ➡ based on model's input_shape(whether extend the feature's dim)
    if model_type['input_shape'] == 'N * L * K * D':
        outputs = model(features)
    elif model_type['input_shape'] == 'N * L * K':
        outputs = model(torch.squeeze(features))
    
    # Calculate the loss based on the model's type(whether to use labels as the ground-truth or it's original features)
    if model_type['type'] == 'Generate':
        loss = criterion(outputs, torch.squeeze(features))  # the reconstruction loss
    elif model_type['type'] == 'Predict':
        loss  = criterion(outputs, labels) # the prediction loss
        
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")