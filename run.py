#!/usr/bin/env python
# coding: utf-8

# Required imports
import sys
import numpy as np
import torch
import lightning.pytorch as pl
from torchinfo import summary
from torchview import draw_graph
import matplotlib.pyplot as plt
import pandas as pd
import torchmetrics

def preprocess(x):
    return (x - np.mean(x)) / np.std(x)

if (torch.cuda.is_available()):
    device = ("cuda")
    print("GPU in use...")
else:
    device = ("cpu")
    print("Only cpu being used...")

data = np.loadtxt("ecg.csv", delimiter=',')

X = data[:,:-1]
X_preprocessed = np.apply_along_axis(preprocess,0,X)
X = X_preprocessed

Y = data[:,-1:]

class MaskedTransformerBlock(torch.nn.Module):
    def __init__(self,
                 latent_size = 64,
                 num_heads = 4,
                 dropout = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.layer_norm1 = torch.nn.LayerNorm(latent_size)
        self.layer_norm2 = torch.nn.LayerNorm(latent_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(latent_size,
                                      latent_size)
        self.mha = torch.nn.MultiheadAttention(latent_size,
                                               num_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.activation = torch.nn.GELU()
        
    def make_causal_mask(self, sz: int):
        return torch.triu(torch.full((sz, sz), True), diagonal=1).to(device)

    def forward(self, x):
        y = x
        y = self.layer_norm1(y)
        y = self.mha(y,y,y,attn_mask=self.make_causal_mask(y.shape[1]))[0]
        x = y = x + y
        y = self.layer_norm2(y)
        y = self.linear(y)
        y = self.activation(y)
        return x + y

class RegressionLightningModule(pl.LightningModule):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.loss = torch.nn.BCEWithLogitsLoss()
    
    def predict(self, x):
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y_true = train_batch
        y_pred = self(x)
        loss = self.loss(y_pred,y_true)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y_true = val_batch
        y_pred = self(x)
        loss = self.loss(y_pred,y_true)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

class RegressionNetwork(RegressionLightningModule):
    def __init__(self,
                 latent_size=140,
                 num_heads=4,
                 n_blocks=4,
                 **kwargs):
        super().__init__(**kwargs)

        self.embedding = torch.nn.Linear(1,latent_size,bias=False) 
        self.transformer_blocks = torch.nn.Sequential(*[
            MaskedTransformerBlock(latent_size=latent_size,
                                   num_heads=num_heads) for _ in range(n_blocks)
        ])
        
        self.AVGpool = torch.nn.AdaptiveAvgPool1d(1)
        self.logit = torch.nn.Linear(latent_size,1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        y = torch.unsqueeze(x,-1)
        y = self.embedding(y.to(device))
        y = self.transformer_blocks(y)
        y = self.AVGpool(y.permute(0,2,1)).permute(0,2,1).squeeze()
        y = self.logit(y)
        return y

    def predict(self, x):
        y = self.forward(x)
        y = self.sigmoid(y)
        return y

# Define a permutation needed to shuffle both
# inputs and target in the same manner...
shuffle = np.random.permutation(X.shape[0])
X_shuffled = X[shuffle,:]
Y_shuffled = Y[shuffle,:]

# Keep 70% for training and remaining for validation
split_point = int(X_shuffled.shape[0] * 0.7)
x_train = X_shuffled[:split_point]
y_train = Y_shuffled[:split_point]
x_val = X_shuffled[split_point:]
y_val = Y_shuffled[split_point:]

model = RegressionNetwork.load_from_checkpoint('./epoch=99-step=11000.ckpt', map_location=torch.device('cpu'))

batch_size = 32
all_data = torch.utils.data.DataLoader(list(zip(torch.Tensor(X[:64]),
                                                torch.Tensor(Y[:64]))),
                                       shuffle=True, batch_size=batch_size,
                                       num_workers=8)

logger = pl.loggers.CSVLogger("lightning_logs_demo",
                              name="Demo-Loading",
                              version="demo-0")

trainer = pl.Trainer(logger=logger,
                     max_epochs=0,
                     enable_progress_bar=True,
                     log_every_n_steps=0,
                     enable_checkpointing=False,
                     callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=1)])

trainer.validate(model,all_data)

# Moving the model to from the GPU (cuda) to the cpu
model = model.to('cpu')
# Changing the device name to 'cpu' so the model uses this device in its methods
device = 'cpu'

# Check shape of the validation data for potential index values to predict
x_val.shape

# Predictions of 0 or 1
# Index must be between 0 and 1499 inclusive
# Change index values to predict specific rows
predictions = model.predict(torch.Tensor(x_val[0:50]).to('cpu'))
predictions = np.rint(predictions.cpu().detach().numpy())

# Actual values
# Change the index values to match the values for the predictions
actual = y_val[0:50]

# Comparing actual values to predictions
np.array_equal(actual, predictions)

# Concatenating actual and prediction arrays for side by side comparison
demo_results = np.concatenate((actual, predictions), axis=1)
demo_results
