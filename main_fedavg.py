import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from client import FLClientFedAvg
from server import Server

NUM_CLIENTS = 10
K = 4
NUM_ROUNDS = 5
NUM_CLASSES_PER_CLIENT = 10

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Check if all classes are represented
    union_classes = np.array([])
    while len(union_classes) < 10:
        client_classes = [
            np.random.choice(np.arange(0, 10), NUM_CLASSES_PER_CLIENT, replace=False)
            for i in range(NUM_CLIENTS)
        ]
        for i in range(NUM_CLIENTS):
            union_classes = np.union1d(union_classes, client_classes[i])

    num_clients_per_class = np.zeros(10)
    for i in range(NUM_CLIENTS):
        for j in range(NUM_CLASSES_PER_CLIENT):
            num_clients_per_class[client_classes[i][j]] += 1
    # print(num_clients_per_class)

    divided_indices_per_class = []
    for i in range(10):
        splits = np.array_split(
            np.where(np.isin(trainset.targets, [i]))[0], num_clients_per_class[i]
        )
        divided_indices_per_class.append(splits)

    # Split trainset into 10 mutually exclusive parts according to classes
    splits_taken = np.zeros(10)
    trainsets = []
    for i in range(NUM_CLIENTS):
        client_samples = []
        for j in client_classes[i]:
            client_samples.append(divided_indices_per_class[j][int(splits_taken[j])])
            splits_taken[j] += 1
        client_samples = np.concatenate(client_samples)
        print(
            "Client {} has the classes: {} totalling to {} samples".format(
                i, client_classes[i], client_samples.shape[0]
            )
        )
        trainsets.append(torch.utils.data.Subset(trainset, client_samples))

    clients = []
    for i in range(NUM_CLIENTS):
        clients.append(FLClientFedAvg(trainsets[i]))

    server = Server(testset)
    for i in range(NUM_ROUNDS):
        print("\n\n ========== Round {} ==========".format(i))
        server.federate(clients)
        server.evaluate()

    #    client_test = FLClient(trainsets[0])
    #    client_test.train(5)
