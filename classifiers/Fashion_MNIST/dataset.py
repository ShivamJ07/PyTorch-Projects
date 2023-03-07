import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataset(batch_size: int):
    train_set = datasets.FashionMNIST(
        root='./dataset', train=True, download=True, transform=transforms.Compose([
            transforms.Resize(24),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2869], std=[0.3346])
        ]))

    test_set = datasets.FashionMNIST(
        root='./dataset', train=False, download=True, transform=transforms.Compose([
            transforms.Resize(24),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2879], std=[0.3341])
        ]))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True
    )

    # image, label = next(iter(train_set))
    # plt.imshow(image.squeeze(), cmap="gray")
    # plt.show()

    # print(get_mean_std(train_loader))
    # print(get_mean_std(test_loader))

    return (train_loader, test_loader)


def check_accuracy(loader, model):
    total_correct = 0
    total_samples = 0
    model.eval()

    with torch.no_grad():
        for data, label in loader:
            # Model will return a number of float values, argmax will return the index of the largest value along the second dimension
            # Recall that the first axis corresponds to batch size, so the second axis corresponds to the model outputs for each batch
            # If batch size is N, model_output will contain N values where each value is the prediction for that batch
            model_output = model(data).argmax(dim=1)
            total_correct += (model_output == label).sum()
            total_samples += model_output.shape[0]

        print(
            f'Correctly classified {(total_correct)}/{total_samples} with an accuracy of {(float(total_correct)/total_samples)*100:.2f}')
