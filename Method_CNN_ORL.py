#here I imported the method.py and Evaluate_Accuract given and also some modules
from method import method
from Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_CNN_ORL(method, nn.Module):
    #i arbitrarily picked to run for 44 epochs with 44 as the batch size, used the same learning rate like in Method_MLP.py
    data = None
    max_epoch = 99
    learning_rate = 1e-3
    batch_size = 140
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        #cnn layers, arbitrarily picked values here as well and used relu activation function
        self.conv1 = nn.Conv2d(1, 44, 4, padding=1)
        self.conv2 = nn.Conv2d(44, 74, 4, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        #fully connected layers
        self.fc1 = nn.Linear(74 * 27 * 22, 128)
        self.fc2 = nn.Linear(128, 40)
    #this pass images through the CNN via forward pass
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        #this flatten the image
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    #this preps orl dataset for the cnn; i reshaped it to match the image size (112 x 92 x 3) of the dataset
    def prepare(self, X):
        #i only used one channel for building cnn and i divided by 255 to normalize the pixel values to a 0-1 range
        X = torch.FloatTensor(np.array(X))[:, :, :, 0] /255.0
        X = X.reshape(-1, 1, 112, 92)
        return X
    #this trains the model
    def train_model(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        #this preps the data
        X = self.prepare(X)
        y = torch.LongTensor(np.array(y)) - 1
        #i used the taining evaluator from the given Evaluate_Accuracy.py
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        #kept track of the training losses for all epochs
        self.loss_list = []
        epoch = 0
        while epoch < self.max_epoch:
            #this shuffles training data
            order = torch.randperm(X.size(0))
            X = X[order]
            y = y[order]
            i = 0
            while i < X.size(0):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += self.batch_size
            #calcs training accuracy
            with torch.no_grad():
                predictions = self.forward(X).max(1)[1]
            accuracy_evaluator.data = {
                'true_y': y.numpy(),
                'pred_y': predictions.numpy()}
            accuracy = accuracy_evaluator.evaluate()
            print('Epoch:', epoch, 'Accuracy:', accuracy, 'Loss:', loss.item())
            #this saves training loss
            self.loss_list.append(loss.item())
            epoch += 1

    #this tests the CNN in batches
    def test_model(self, X):
        X = self.prepare(X)
        predictions = []
        i = 0
        with torch.no_grad():
            while i < X.size(0):
                pred = self.forward(X[i:i + 1000]).max(1)[1]
                predictions.append(pred)
                i += 1000
        return torch.cat(predictions)

    #this runs the cnn
    def run(self):
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        pred_y = self.test_model(self.data['test']['X'])
        return {
            'pred_y': pred_y,
            'true_y': np.array(self.data['test']['y']) - 1}