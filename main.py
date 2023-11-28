
if __name__ == '__main__':
    import os
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from ml_framework import system_check
    from ml_framework import load_data
    from ml_framework import get_device
    from ml_framework import train
    from ml_framework import test
    from ml_models import NeuralNetwork

    system_check()

    path_input = "data/input/"
    path_output = "data/output/"

    # Getting Data and create data loaders.
    training_data, test_data = load_data(path_input=path_input)

    batch_size = 64

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Training model
    device = get_device()
    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")


    model_name = "model.pth"

    torch.save(model.state_dict(), os.path.join(path_output, model_name))
    print("Saved PyTorch Model State")

    # Using model
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(os.path.join(path_output, model_name)))

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')




