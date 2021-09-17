import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import numpy
from pytorch.util.nn_tools import simple_layer_creator, get_output_shape

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# define a similarity for comparing distributions of treated/control representations
# Here, I just considered Euclidean distance in the representation space
# (I think this is just what they do in the paper?)
def similarity(o1, o2):
    centroid1 = torch.mean(o1, dim=0)
    centroid2 = torch.mean(o2, dim=0)

    dist = torch.dist(centroid1, centroid2)

    return dist


# The Balancing Neural Network - Johansson et al. ICML '16
class BalancingNeuralNetwork(nn.Module):
    def __init__(self,
                 n_features=None,
                 n_outputs=1,
                 pre_treatment=None,
                 post_treatment=None,
                 optimizer=optim.SGD,
                 optim_params=None,
                 pred_loss=nn.MSELoss,
                 sim_loss=similarity,
                 verbose=False):
        # super(BalancingNeuralNetwork, self).__init__()
        super().__init__()

        # Using sequential here, but can define pre-treatment as a class or multiple layers!
        # The "pre_treatment" layers give us the representation before adding the indicator for treatment
        # We concatenate the output here with the treatment indicator later and pass them to the similarity and the
        # final layers
        self.pre_treatment_rep_size = 10
        if pre_treatment is not None:
            if isinstance(pre_treatment, dict):
                pre_treatment = simple_layer_creator(layer_dict=pre_treatment)
            elif isinstance(pre_treatment, list):
                pre_treatment = simple_layer_creator(layer_list=pre_treatment)
            self.pre_treatment = pre_treatment
            self.pre_treatment_rep_size = get_output_shape(pre_treatment)
        elif n_features is not None:
            self.n_features = n_features
            self.pre_treatment_rep_size = 25
            self.pre_treatment = nn.Sequential(
                nn.Linear(n_features, 25),
                nn.ReLU(),  # if we want activation
                nn.Linear(25, self.pre_treatment_rep_size),
                nn.ReLU(),
            )
        else:
            print("No pre_treatment layer or n_features defined!")

        # the first layer of the "treat_concat" appends the treatment, so +1 to output of pre-treatment representation
        if post_treatment is not None:
            if isinstance(post_treatment, dict):
                post_treatment = simple_layer_creator(layer_dict=post_treatment)
            elif isinstance(pre_treatment, list):
                post_treatment = simple_layer_creator(layer_list=post_treatment)
            self.post_treatment = post_treatment
        else:
            self.post_treatment = nn.Sequential(
                nn.Linear(self.pre_treatment_rep_size + 1, 25),
                nn.ReLU(),
                nn.Linear(25, 25),
                nn.ReLU(),
                nn.Linear(25, n_outputs)
            )

        # defining optimizers
        if optim_params is not None:
            self.optimizer = optimizer(self.parameters(), **optim_params)
        else:
            self.optimizer = optimizer(self.parameters(), lr=0.001)

        # defining loss functions
        self.pred_loss = pred_loss()
        self.sim_loss = sim_loss

        # verbosity and other helpers
        self.verbose = verbose

        # just to keep track of the number of epochs, not really necessary
        self.epoch = 0

    def forward(self, x, t):
        # keep track of all outputs I need
        # for BNN this is mainly the final output for prediction loss
        output = {}

        # pass through the first "pre-treatment" layers
        x = self.pre_treatment(x)
        output["pre_treatment_rep"] = x
        # another option for ReLU is
        # x = F.ReLU(x)

        # concatenate treatment
        second_stage_input = torch.cat((x, t), 1)

        # then pass through final layers to get predictions
        final_output = self.post_treatment(second_stage_input)
        output["y"] = final_output

        return output

    # this "backward" function is just me telling which losses/how to compute them.
    # the real "backward" is all in the "loss.backward()" call from PyTorch
    def backward(self, output, y, t):

        # get the predictions
        y_pred = output["y"]

        # concat the treatment to pre-treatment representation
        pre_treat_rep = output["pre_treatment_rep"]
        treat_rep_concat = torch.cat((pre_treat_rep, t), 1)

        # need zero_grad before every update I think?
        self.optimizer.zero_grad()

        # calculate the prediction loss, here it is MSELoss
        pred_loss = self.pred_loss(y_pred, y)

        # Here I am calculating the similarity between treated and control representations
        # For the similarity function above, I only use euclidean distance on the centroids. Probably there is a better
        # way but I think the idea here is enough
        treated = t.view(-1) == 1
        control = t.view(-1) == 0
        sim_loss = self.sim_loss(treat_rep_concat[treated],
                                 treat_rep_concat[control])

        # Combine the losses
        loss = pred_loss + sim_loss
        # loss = pred_loss

        if self.verbose:
            print(self.epoch, loss.item())

        # Backprop
        loss.backward()

        # Step the optimizer
        self.optimizer.step()

    def fit(self, x, y, t, epochs=1000, verbose=False):
        if type(x) == numpy.ndarray:
            x = torch.from_numpy(x).float()
        if type(y) == numpy.ndarray:
            y = torch.from_numpy(y.reshape(-1, 1)).float()
        if type(t) == numpy.ndarray:
            t = torch.from_numpy(t.reshape(-1, 1)).float()
        self.verbose = verbose
        for epoch in range(epochs):
            self.epoch = epoch
            output = self.forward(x, t)
            self.backward(output, y, t)

    def predict(self, x):
        if type(x) == numpy.ndarray:
            x = torch.from_numpy(x).float()
        treat_pred = self.forward(x, torch.ones(x.shape[0], 1))["y"]
        cont_pred = self.forward(x, torch.zeros(x.shape[0], 1))["y"]

        pred = treat_pred - cont_pred
        pred = pred.detach().numpy()

        return pred

    def forward_numpy(self, x, t):
        npt = t.copy()
        if type(x) == numpy.ndarray:
            x = torch.from_numpy(x).float()
        if type(t) == numpy.ndarray:
            t = torch.from_numpy(t.reshape(-1, 1)).float()
        output = self.forward(x, t)
        # output0 = self.forward(x, t0)

        y = output["y"].detach().numpy().reshape(-1)

        y1 = y[npt == 1]
        y0 = y[npt == 0]

        return y1, y0
