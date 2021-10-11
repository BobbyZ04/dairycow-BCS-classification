# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:54:23 2021

@author: BOZ
"""

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from pytorchtools import EarlyStopping

class Trainer:
    def __init__(self, model, optimizer, criterion, train_x, train_y, val_x, val_y, patience):
                self.model = model
                self.optimizer = optimizer
                self.criterion = criterion
                self.train_x = train_x
                self.train_y = train_y
                self.val_x = val_x
                self.val_y = val_y
                self.patience = patience

    def train_val(self, n_epochs, batch_size):
        train_losses = []
        train_accs = []
        validation_losses = []
        validation_accs = []
        
        early_stopping = EarlyStopping(patience = self.patience, verbose=True)
        
        for epoch in range(1, n_epochs+1):
            #train_loss = 0.0
            permutation = torch.randperm((self.train_x).size()[0])
            training_loss = []
            prediction = []
            target = []
            accuracy = []
            
            
            for i in tqdm(range(0,(self.train_x).size()[0], batch_size)):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = self.train_x[indices], self.train_y[indices]
                #if torch.cuda.is_available():
                #   batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                #   batch_x = batch_x.type(torch.float)
                batch_y = batch_y.type(torch.LongTensor)
                self.optimizer.zero_grad()
                #torch.cuda.empty_cache()

                outputs = self.model(batch_x)
                loss = self.criterion(outputs,batch_y)

                #softmax layer for prediction
                softmax = torch.exp(outputs)
                prob = list(softmax.detach().numpy())
                predictions = np.argmax(prob, axis=1)
                prediction.append(predictions)
                target.append(batch_y)
                
                training_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
 
            training_loss = np.average(training_loss)
    
            #calculate training accuracy
            for i in range(len(prediction)):
                accuracy.append(accuracy_score(target[i],prediction[i]))  
            
            #calculate validation loss and accuracy
            output_test = self.model(self.val_x)
            val_y = self.val_y.type(torch.LongTensor)
            val_loss = self.criterion(output_test,val_y)
            val_loss = val_loss.item()
            
            softmax_test = torch.exp(output_test)
            prob_test = list(softmax_test.detach().numpy())
            predictions_test = np.argmax(prob_test, axis=1)
            val_acc = accuracy_score(val_y, predictions_test)
            
            #output metrics and performance
            train_losses.append(training_loss)
            train_accs.append(np.average(accuracy))
            
            validation_losses.append(val_loss)
            validation_accs.append(np.average(val_acc))
            
            #early stopping
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            print('epoch:', epoch, '  training loss: ', training_loss,'  training accuracy: ', format(np.average(accuracy),'.5%'))
            print('\t validation loss: ', val_loss,  '\t validation accuracy: ', format(np.average(val_acc),'.5%'))
            
        return train_losses, train_accs, validation_losses, validation_accs, predictions_test