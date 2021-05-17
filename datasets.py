from torchvision import datasets


class MNIST:
    def __init__(self, root, transform):
        self.num_channels = 1

        self.train = datasets.MNIST(root, train=True, download=True, transform=transform)
        self.test = datasets.MNIST(root, train=False, download=True, transform=transform)


class FashionMNIST:
    def __init__(self, root, transform):
        self.num_channels = 1

        self.train = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
        self.test = datasets.FashionMNIST(root, train=False, download=True, transform=transform)


class CIFAR10:
    def __init__(self, root, transform):
        self.num_channels = 3

        self.train = datasets.CIFAR10(root, train=True, download=True, transform=transform)
        self.test = datasets.CIFAR10(root, train=False, download=True, transform=transform)
