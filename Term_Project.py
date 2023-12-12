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

data

data.shape

type(data[0][-1])

data

X = data[:,:-1]
X_preprocessed = np.apply_along_axis(preprocess,0,X)
X = X_preprocessed
X

X.shape

Y = data[:,-1:]

Y.shape

XY.shape

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

    # No complications with regression...
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
                 latent_size=128,
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

model = RegressionNetwork()
summary(model,input_size=x_train[0:5].shape,device=device)

model_graph = draw_graph(model, input_size=x_train[0:5].shape, device=device,
                        hide_inner_tensors=True,hide_module_functions=True,
                        expand_nested=False, depth=3)
model_graph.visual_graph

batch_size = 32
xy_train = torch.utils.data.DataLoader(list(zip(torch.Tensor(x_train),\
                                                torch.Tensor(y_train))),\
                                       shuffle=True, batch_size=batch_size)
xy_val = torch.utils.data.DataLoader(list(zip(torch.Tensor(x_val),\
                                              torch.Tensor(y_val))),\
                                     shuffle=False, batch_size=batch_size)

logger = pl.loggers.CSVLogger("lightning_logs",
                              name="Term-Project",
                              version="demo-0")

trainer = pl.Trainer(max_epochs=100,
                     logger=logger,
                     enable_progress_bar= True,
                     log_every_n_steps=0,
                     enable_checkpointing=True,
                     callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=1)])

preliminary_result = trainer.validate(model, dataloaders=xy_val)

trainer.fit(model, train_dataloaders=xy_train, val_dataloaders=xy_val)

final_result = trainer.validate(model, dataloaders=xy_val)

logger.log_dir + "/metrics.csv"

# results = pd.read_csv("logs/Term-Project/version_0/metrics.csv", delimiter=',')
results = pd.read_csv(logger.log_dir + "/metrics.csv", delimiter=',')
results

plt.plot(results["epoch"][np.logical_not(np.isnan(results["train_loss"]))],\
         results["train_loss"][np.logical_not(np.isnan(results["train_loss"]))],\
         label="Training")
plt.plot(results["epoch"][np.logical_not(np.isnan(results["val_loss"]))],\
         results["val_loss"][np.logical_not(np.isnan(results["val_loss"]))],\
         label="Validation")

plt.legend()
plt.ylabel("BCE Loss")
plt.xlabel("Epoch")
plt.savefig("./term-project-val-loss.png")
plt.show()


print("Validation loss:",*["%.8f"%(x) for x in results['val_loss'][np.logical_not(np.isnan(results["val_loss"]))][0::10]])

