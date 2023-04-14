import torch
import wandb


def validate(model, val_loader, device):
    model.eval()

    # Run the model on some validate examples
    with torch.no_grad():
        
        correct, total = 0, 0
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})

    # Save the model in the exchangeable ONNX format
    # torch.onnx.export(model, features, "model.onnx")
    # wandb.save("model.onnx")