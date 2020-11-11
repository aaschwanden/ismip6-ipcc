#!/usr/bin/env python

# Copyright (C) 2019-20 Andy Aschwanden

from glob import glob
import matplotlib.lines as mlines
from netCDF4 import Dataset as NC
import numpy as np
import os
import pylab as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm

import torch
import theano as tt


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


hist_start = 2008
hist_end = 2014
proj_start = hist_end + 1
proj_end = 2100

# Greenland only though this could easily be extended to Antarctica
domain = {"GIS": "greenland_mass_200204_202008.txt"}

for d, data in domain.items():
    print(f"Analyzing {d}")

    # Load the GRACE data
    grace = pd.read_csv(
        data, header=30, delim_whitespace=True, skipinitialspace=True, names=["Time", "Mass (Gt)", "Sigma (Gt)"]
    )
    # Normalize GRACE signal to the starting date of the projection
    grace["Mass (Gt)"] -= np.interp(proj_start, grace["Time"], grace["Mass (Gt)"])

    # Get the GRACE trend
    grace_time = (grace["Time"] >= hist_start) & (grace["Time"] <= proj_start)
    grace_hist_df = grace[grace_time]
    x = grace_hist_df["Time"]
    y = grace_hist_df["Mass (Gt)"][(grace["Time"] >= hist_start) & (grace["Time"] <= proj_start)]
    s = grace_hist_df["Sigma (Gt)"][(grace["Time"] >= hist_start) & (grace["Time"] <= proj_start)]
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    p = ols.params
    grace_bias = p[0]
    grace_trend = p[1]

    x_train = x.astype(np.float32).values.reshape(-1, 1)
    y_train = y.astype(np.float32).values.reshape(-1, 1)
    s_train = s.astype(np.float32).values.reshape(-1, 1)

    x_train -= x_train.mean()
    y_train -= y_train.mean()
    s_train -= s_train.mean()

    inputDim = 1  # takes variable 'x'
    outputDim = 1  # takes variable 'y'
    learningRate = 0.1
    epochs = 100

    model = LinearRegressionModel(inputDim, outputDim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = torch.Tensor(torch.from_numpy(x_train).cuda())
            labels = torch.Tensor(torch.from_numpy(y_train).cuda())
        else:
            inputs = torch.Tensor(torch.from_numpy(x_train))
            labels = torch.Tensor(torch.from_numpy(y_train))

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        print(loss)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        print("epoch {}, loss {}".format(epoch, loss.item()))

    with torch.no_grad():
        torch_trend = model.linear.weight.numpy()[0][0]

    print(f"Statsmodel OLS trend: {grace_trend:.1f}, Torch trend {torch_trend:.1f}")

    # https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/
