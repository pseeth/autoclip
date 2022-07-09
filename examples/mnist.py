"""
Adapted from the pytorch mnist example found at https://github.com/pytorch/examples/blob/main/mnist/main.py
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import autoclip
from autoclip.torch import QuantileClip


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    clipper: autoclip.torch.Clipper,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    device: torch.device = torch.device("cuda"),
    log_interval: int = 10,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        clipper.step()
        optimizer.step()
        lr_scheduler.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cuda"),
):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    config = {"num_epochs": 14, "max_learning_rate": 1e-3, "weight_decay": 0.05}
    train_kwargs = {"batch_size": 64}
    test_kwargs = {"batch_size": 1000}
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    else:
        device = torch.device("cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["max_learning_rate"],
        weight_decay=config["weight_decay"],
    )
    clipper = QuantileClip(
        model.parameters(),
        quantile=0.8,
        history_length=1000,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["max_learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=config["num_epochs"],
    )

    for epoch in range(1, config["num_epochs"] + 1):
        train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            clipper=clipper,
            lr_scheduler=scheduler,
            epoch=epoch,
            device=device,
            log_interval=10,
        )
        test(model=model, test_loader=test_loader, device=device)

    torch.save(model.state_dict(), "mnist_cnn.pth")


if __name__ == "__main__":
    main()
