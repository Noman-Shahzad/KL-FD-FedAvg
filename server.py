import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import matplotlib.pyplot as plt

class ServerNet(nn.Module):
    def __init__(self):
        super(ServerNet, self).__init__()
        self.cv1 = nn.Conv2d(3, 32, 5)
        self.cv2 = nn.Conv2d(32, 64, 5)
        self.cv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.cv1(x))
        x = self.pool(F.relu(self.cv2(x)))
        x = self.pool(F.relu(self.cv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Server:
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, batch_size=32, shuffle=True
        )
        self.model = ServerNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.communication_rounds = 0
        self.total_data_transferred = 0
        self.metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'loss': [],
            'time': []
        }

    def aggregate(self, client_updates):
        for cu in client_updates:
            for key in cu:
                cu[key] *= 1 / len(client_updates)
        self.model.load_state_dict(client_updates[0])
        for cu in client_updates[1:]:
            for key in cu:
                self.model.state_dict()[key] += cu[key]

    def federate(self, clients):
        updates = []
        start_time = time.time()
        for c in clients:
            print("\n -----> Training client {}".format(clients.index(c)))
            c_model = ServerNet()
            c_model.to(self.device)
            c_model.load_state_dict(self.model.state_dict())
            c.train(5, c_model)
            updates.append(c.get_weights())
            self.total_data_transferred += sum(p.numel() for p in c_model.parameters())
        self.aggregate(updates)
        self.communication_rounds += 1
        end_time = time.time()
        self.metrics['time'].append(end_time - start_time)
        print(f"Round {self.communication_rounds} completed in {end_time - start_time:.2f} seconds")

    def evaluate(self):
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        accuracy = 100 * correct / total
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        avg_loss = running_loss / len(self.test_loader)

        self.metrics['accuracy'].append(accuracy)
        self.metrics['precision'].append(precision)
        self.metrics['recall'].append(recall)
        self.metrics['f1_score'].append(f1)
        self.metrics['loss'].append(avg_loss)

        print(f"========== !!!!! Evaluation Metrics !!!!! ==========")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Total Communication Rounds: {self.communication_rounds}")
        print(f"Total Data Transferred: {self.total_data_transferred / 1e6:.2f} MB")

    def plot_metrics(self):
        rounds = range(1, self.communication_rounds + 1)
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(rounds, self.metrics['accuracy'], label='Accuracy')
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Rounds')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(rounds, self.metrics['precision'], label='Precision')
        plt.xlabel('Rounds')
        plt.ylabel('Precision')
        plt.title('Precision over Rounds')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(rounds, self.metrics['recall'], label='Recall')
        plt.xlabel('Rounds')
        plt.ylabel('Recall')
        plt.title('Recall over Rounds')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(rounds, self.metrics['f1_score'], label='F1 Score')
        plt.xlabel('Rounds')
        plt.ylabel('F1 Score')
        plt.title('F1 Score over Rounds')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def get_weights(self):
        return self.model.state_dict()