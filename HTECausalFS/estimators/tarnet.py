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


class TARNet(nn.Module):
    def __init__(self,
                 n_features=None,
                 n_outputs=1,
                 pre_treatment=None,
                 post_treatment_treated=None,
                 post_treatment_control=None,
                 post_treatment=None,
                 optimizer=optim.SGD,
                 optim_params=None,
                 pred_loss=nn.MSELoss,
                 verbose=False):
        # super(TARNet, self).__init__()
        super().__init__()

        if post_treatment is not None and (post_treatment_control is None or post_treatment_treated is None):
            post_treatment_control = post_treatment
            post_treatment_treated = post_treatment

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
            self.pre_treatment = nn.Sequential(
                nn.Linear(n_features, 25),
                nn.ReLU(),  # if we want activation
                nn.Linear(25, self.pre_treatment_rep_size),
                nn.ReLU(),
            )
        else:
            print("No pre_treatment layer or n_features defined")

        if post_treatment_treated is not None:
            if isinstance(post_treatment_treated, dict):
                post_treatment_treated = simple_layer_creator(layer_dict=post_treatment_treated)
            elif isinstance(pre_treatment, list):
                post_treatment_treated = simple_layer_creator(layer_list=post_treatment_treated)
            self.post_concat_treated = post_treatment_treated
        else:
            # the first layer of the "treat_concat" appends the treatment,
            # so +1 to output of pre-treatment representation
            self.post_concat_treated = nn.Sequential(
                nn.Linear(self.pre_treatment_rep_size + 1, 25),
                nn.ReLU(),
                nn.Linear(25, 25),
                nn.ReLU(),
                nn.Linear(25, n_outputs)
            )
        if post_treatment_control is not None:
            if isinstance(post_treatment_control, dict):
                post_treatment_control = simple_layer_creator(layer_dict=post_treatment_control)
            elif isinstance(pre_treatment, list):
                post_treatment_control = simple_layer_creator(layer_list=post_treatment_control)
            self.post_concat_control = post_treatment_control
        else:
            self.post_concat_control = nn.Sequential(
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
        self.pred_loss_t = pred_loss()
        self.pred_loss_c = pred_loss()

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

        # separate treated and control
        treated = t.view(-1) == 1
        control = t.view(-1) == 0

        # separate outputs for treated and control
        yt = self.post_concat_treated(second_stage_input[treated, :])
        yc = self.post_concat_control(second_stage_input[control, :])

        output["yt"] = yt
        output["yc"] = yc

        return output

    def backward(self, output, y, t):
        # get the predictions
        y_pred_t = output["yt"]
        y_pred_c = output["yc"]

        treated = t.view(-1) == 1
        control = t.view(-1) == 0

        self.optimizer.zero_grad()
        # calculate the prediction loss, here it is MSELoss
        pred_loss_t = self.pred_loss_t(y_pred_t, y[treated])
        pred_loss_c = self.pred_loss_c(y_pred_c, y[control])

        loss = pred_loss_t + pred_loss_c

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
        t = torch.cat((torch.ones(x.shape[0], 1), torch.zeros(x.shape[0], 1)))
        x = x.repeat(2, 1)
        pred = self.forward(x, t)

        treat_pred = pred["yt"]
        cont_pred = pred["yc"]

        pred = treat_pred - cont_pred
        pred = pred.detach().numpy().reshape(-1)

        return pred
