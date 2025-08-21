import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import Net
from server import ServerNet

class FLClient:
    def __init__(self, train_data):
        self.train_data = train_data
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=32, shuffle=True
        )
        num_conv_layers = np.random.randint(1, 3)
        num_fc_layers = np.random.randint(2, 5)
        conv_param_list = []
        conv_param_list.append([3, 32, 5])
        filters = 32
        conv_output_size = 14
        for i in range(1, num_conv_layers):
            conv_param_list.append([filters, filters * 2, 5])
            filters *= 2
            conv_output_size = (conv_output_size - 4) // 2
        fc_param_list = []
        units = 128
        fc_param_list.append([filters * conv_output_size * conv_output_size, units])
        for i in range(1, num_fc_layers - 1):
            fc_param_list.append([units, units // 2])
            units //= 2
        fc_param_list.append([units, 10])
        self.model = Net(conv_param_list, fc_param_list)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, num_epochs, server_model):
        self.server_model_copy = copy.deepcopy(server_model)
        self.server_model_copy.to(self.device)
        # Distill local model knowledge to server model.
        print("\tDistilling local model knowledge to server model.")
        kld_loss = nn.KLDivLoss(reduction="batchmean")
        ce_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.server_model_copy.parameters(), lr=0.001)
        self.model.eval()
        self.server_model_copy.train()
        temperature = 2.0  # Temperature scaling
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    local_model_logits = self.model(inputs)
                server_model_logits = self.server_model_copy(inputs)
                distillation_loss = kld_loss(
                    F.log_softmax(server_model_logits / temperature, dim=1),
                    F.softmax(local_model_logits / temperature, dim=1),
                ) * (temperature ** 2)
                clf_loss = ce_loss(server_model_logits, labels)
                loss = distillation_loss + clf_loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(
                "\t\tEpoch %d loss: %.3f"
                % (epoch + 1, running_loss / len(self.train_loader))
            )

        # Distill server model knowledge to local model.
        server_model.to(self.device)
        print("\tDistilling server model knowledge to local model.")
        kld_loss = nn.KLDivLoss(reduction="batchmean")
        ce_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        server_model.eval()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    server_model_logits = server_model(inputs)
                local_model_logits = self.model(inputs)
                distillation_loss = kld_loss(
                    F.log_softmax(local_model_logits / temperature, dim=1),
                    F.softmax(server_model_logits / temperature, dim=1),
                ) * (temperature ** 2)
                clf_loss = ce_loss(local_model_logits, labels)
                loss = distillation_loss + clf_loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(
                "\t\tEpoch %d loss: %.3f"
                % (epoch + 1, running_loss / len(self.train_loader))
            )

        # Finetune distilled local model on local data.
        self.model.to(self.device)
        print("\tFinetuning distilled local model on local data.")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for epoch in range(3):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(
                "\t\tEpoch %d loss: %.3f"
                % (epoch + 1, running_loss / len(self.train_loader))
            )

    def get_weights(self):
        return self.server_model_copy.state_dict()


class FLClientFedAvg:
    def __init__(self, train_data):
        self.train_data = train_data
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=32, shuffle=True
        )

        self.model = ServerNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, num_epochs, server_model):
        # Finetune distilled local model on local data.
        self.model.to(self.device)
        self.model.load_state_dict(server_model.state_dict())
        print("\tFinetuning distilled local model on local data.")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for epoch in range(3):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(
                "\t\tEpoch %d loss: %.3f"
                % (epoch + 1, running_loss / len(self.train_loader))
            )

    def get_weights(self):
        return self.model.state_dict()
    
    