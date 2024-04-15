import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from PIL import Image
from sklearn.model_selection import train_test_split

class PathfinderModel(nn.Module):
    W = 200
    H = 200

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 50 * 50, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 50 * 50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x
    
class EuclideanLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target):
        d_a = torch.sqrt((inputs[0] - target[0])**2 + (inputs[1] - target[1])**2)
        d_b = torch.sqrt((inputs[2] - target[2])**2 + (inputs[3] - target[3])**2)

        return (d_a + d_b) / 2
    
class Pathfinder:
    def __init__(self) -> None:
        self.model = PathfinderModel()
        self.model_is_trained = False

    def load_data(self, images_path: str, labels_path: str):
        labels = pd.read_csv(labels_path, header=None, index_col=None)
        x = []
        y = []
        for (_, row) in labels.iterrows():
            img = Image.open(os.path.join(images_path, row[0]))
            img = torchvision.transforms.functional.pil_to_tensor(img) / 255
            
            x.append(img)
            y.append(torch.tensor(row[1:5].to_numpy(dtype=np.float32)))
        
        x = torch.stack(x)
        y = torch.stack(y)

        return (x, y)

    def train(self, images_folder: str="images/", labels_file: str="labels.csv", save_model_path: None | str="model.pt", val_size: float=0.2, epochs: int =2, print_interval: None | int=10):
        x, y = self.load_data(images_folder, labels_file)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size)

        loss_fn = EuclideanLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in range(epochs):
            train_loss = 0.0
            self.model.train()
            for i in range(len(x_train)):
                inputs, labels = x_train[i].unsqueeze(0), y_train[i]

                optimizer.zero_grad()

                output = self.model(inputs)[0]
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                if print_interval is not None and i % print_interval == 0:
                    print(f"Train Epoch {epoch + 1}, Step {i + 1}, Loss: {loss.item()}, Predicted: {output}, Actual: {labels}")

            print(f'Epoch {epoch + 1}, Training loss: {train_loss/len(x_train)}')

            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for i in range(len(x_val)):
                    inputs, labels= x_val[i].unsqueeze(0), y_val[i]
                    output = self.model(inputs)[0]
                    loss = loss_fn(output, labels)
                    val_loss += loss.item()

                    if print_interval is not None and i % print_interval == 0:
                        print(f"Validation Epoch {epoch + 1}, Step {i + 1}, Loss: {loss.item()}, Predicted: {output}, Actual: {labels}")

            print(f'Epoch {epoch + 1}, Validation loss: {val_loss/len(x_val)}')

        if save_model_path is not None:
            torch.save(self.model.state_dict(), save_model_path)

        self.model_is_trained = True

    def detect_line(self, img, model_path="model.pt"):
        if not self.model_is_trained:
            self.model.load_state_dict(torch.load(model_path))

        img = torch.tensor(img, dtype=torch.float32)
        img = img.view(1, 1, 200, 200)
        return self.model(img)[0].numpy(force=True).astype("int32")

def init():
    return Pathfinder()

def main():
    pf = init()
    pf.train(epochs=10, print_interval=None)

if __name__ == "__main__":
    #main()
    pass
