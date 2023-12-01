from torchvision import datasets
from torchvision.transforms import ToTensor

def download_data(path_input):
    # Download training data from open datasets.
    datasets.FashionMNIST(
        root=path_input,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    datasets.FashionMNIST(
        root=path_input,
        train=False,
        download=True,
        transform=ToTensor(),
    )

if __name__ == '__main__':
    # the path should be outside the future container,
    # otherwise the files will be integrated into the container during container building.
    path = f'/path/to/workdir_pqtapp/input'
    download_data(path)