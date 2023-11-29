import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def system_check():
    """
    Prints available resources.
    :return: None
    """
    import torch
    import os

    print('Total CPUs (threads) found:',os.cpu_count())
    print('Available threads for torch:',torch.get_num_threads())

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print('Total GPUs found:',str(count))
        for i in range(torch.cuda.device_count()):
            print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i))
    else:
        print('No GPUs found!')

def load_data(path_input):
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root=path_input,
        train=True,
        download=False,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root=path_input,
        train=False,
        download=False,
        transform=ToTensor(),
    )
    return training_data, test_data

def get_device():
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")