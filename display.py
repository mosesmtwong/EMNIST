import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F


mapping = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F",
    16: "G",
    17: "H",
    18: "I",
    19: "J",
    20: "K",
    21: "L",
    22: "M",
    23: "N",
    24: "O",
    25: "P",
    26: "Q",
    27: "R",
    28: "S",
    29: "T",
    30: "U",
    31: "V",
    32: "W",
    33: "X",
    34: "Y",
    35: "Z",
    36: "a",
    37: "b",
    38: "d",
    39: "e",
    40: "f",
    41: "g",
    42: "h",
    43: "n",
    44: "q",
    45: "r",
    46: "t",
}

# Define the transform to be applied to the MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1736,), (0.3248,))]
)

# Load the MNIST dataset
train_dataset = datasets.EMNIST(
    root="data", split="balanced", train=True, download=True, transform=transform
)
test_dataset = datasets.EMNIST(
    root="data", split="balanced", train=False, download=True, transform=transform
)

train_dataset.data = F.affine(F.hflip(train_dataset.data), -90, (0, 0), 1, 0)
test_dataset.data = F.affine(F.hflip(test_dataset.data), -90, (0, 0), 1, 0)


# Define the function to display MNIST samples
def display_mnist_samples(dataset, num_samples=9):
    # Get a batch of samples from the dataset
    samples, labels = next(
        iter(torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True))
    )

    # Create a figure with a 3x3 grid
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    # Display the samples
    for i, ax in enumerate(axs.flat):
        ax.imshow(samples[i][0], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Label: {mapping[int(labels[i])]}")

    plt.show()


# Display 9 samples from the training dataset
display_mnist_samples(train_dataset)

# Display 9 samples from the test dataset
display_mnist_samples(test_dataset, num_samples=9)
