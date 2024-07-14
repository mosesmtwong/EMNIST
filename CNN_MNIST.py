import time
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import v2

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
EPOCH = 50
BATCH_SIZE = 300
LEARNING_RATE = 0.001


def main():
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.1736,), (0.3248,)),
                transforms.RandomApply(
                    torch.nn.ModuleList(
                        [
                            v2.RandomAffine(
                                degrees=40, translate=(0.3, 0.3), scale=(0.6, 1.1)
                            ),
                            v2.GaussianNoise(0, 0.1),
                        ]
                    ),
                    p=0.6,
                ),
            ]
        ),
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.1736,), (0.3248,)),
                # v2.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.7, 1.1)),
            ]
        ),
    )

    train_dataloader = DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, BATCH_SIZE, shuffle=True)

    model = NeuralNetwork()
    # model.load_state_dict(torch.load(r"models/MNIST/MNIST_v2_50epoch.pth"))
    model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    t1 = time.time()
    for i in range(EPOCH):
        print(f"Epoch {i+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_func, optimizer)
        test_loop(test_dataloader, model, loss_func)
        if i % 50 == 0:
            torch.save(model.state_dict(), rf"models/MNIST/MNIST_{i}epoch.pth")
    print("Done!")
    t2 = time.time()

    print(f"Finished in {t2-t1} seconds")

    torch.save(model.state_dict(), r"models/MNIST/MNIST_100epoch.pth")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 47),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        # print("input:", x.size())
        # for layer in self.CNN:
        #     x = layer(x)
        #     print(layer, x.size())
        #     return x
        return self.CNN(x)


def train_loop(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_func(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}   [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    test_loss, correct = 0, 0

    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_func(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batch
    correct /= size
    print(
        f"test error:\nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":
    main()
