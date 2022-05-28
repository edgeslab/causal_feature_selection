from HTECausalFS.local_fs.pc_fs import PCFeatureSelect
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


class StructureFitEstimator:

    def __init__(self, hte_estimator, learner_econ=False, learner_nn=False, pc_alg="PCDbyPCD", edge_alg="reci",
                 adjust_method="nonforbidden",
                 use_propensity=False, propensity_model=LogisticRegression, propensity_params=None, alpha=0.05,
                 binary_data=False, check_colliders=False, print_adjustment_sets=False,
                 dl_layers=None, dl_params=None):
        self.PCFeatureSelect = PCFeatureSelect(pc_alg=pc_alg, edge_alg=edge_alg, adjust_method=adjust_method,
                                               use_propensity=use_propensity, propensity_model=propensity_model,
                                               propensity_params=propensity_params, alpha=alpha,
                                               binary_data=binary_data, check_colliders=check_colliders)
        self.HTE_Estimator = hte_estimator

        self.learner_econ = learner_econ
        self.learner_nn = learner_nn
        self.dl_layers = dl_layers
        self.dl_params = dl_params
        self.adjustment_set = None
        self.print_adjustment_sets = print_adjustment_sets

        self.temp_hte_estimator = None

    def find_adjustment_set(self, df, adjustment_set=None):
        if adjustment_set is not None:
            self.adjustment_set = adjustment_set
        else:
            self.adjustment_set = self.PCFeatureSelect.get_adjustment_set(df)

        # if len(self.adjustment_set) < 1:
        #     # temp fix if adjustment set is empty
        #     self.adjustment_set = [i for i in df.columns if i != "y" and i != "t"]

    def fit(self, x, y, t, adjustment_set=None):

        dat_dict = dict()
        for j in range(x.shape[1]):
            dat_dict[f"x{j}"] = x[:, j]
        dat_dict["t"] = t
        dat_dict["y"] = y
        df = pd.DataFrame(dat_dict)

        self.find_adjustment_set(df, adjustment_set)

        if self.print_adjustment_sets:
            print(self.adjustment_set)

        if len(self.adjustment_set) == 0:
            new_x = np.zeros((df.shape[0], 1))
        else:
            new_x = df[self.adjustment_set].values

        if len(new_x.shape) < 2:
            new_x = new_x.reshape(-1, 1)

        if self.learner_econ:
            self.HTE_Estimator.fit(y, t, X=new_x)
        elif self.learner_nn:
            # temp_learner = learner(n_features=x_train.shape[1])
            pre_treatment, post_treatment = self.dl_layers(new_x.shape[1])
            params = self.dl_params
            self.temp_hte_estimator = self.HTE_Estimator(pre_treatment=pre_treatment, post_treatment=post_treatment,
                                                         **params)
            self.temp_hte_estimator.fit(new_x, y, t)
        else:
            self.HTE_Estimator.fit(new_x, y, t)

    def predict(self, x):

        dat_dict = dict()
        for j in range(x.shape[1]):
            dat_dict[f"x{j}"] = x[:, j]
        df = pd.DataFrame(dat_dict)

        if len(self.adjustment_set) == 0:
            new_x = np.zeros((df.shape[0], 1))
        else:
            new_x = df[self.adjustment_set].values
        if len(new_x.shape) < 2:
            new_x = new_x.reshape(-1, 1)

        if self.learner_econ:
            pred = self.HTE_Estimator.effect(new_x)
        elif self.learner_nn:
            pred = self.temp_hte_estimator.predict(new_x)
        else:
            pred = self.HTE_Estimator.predict(new_x)

        return pred
