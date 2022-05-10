from HTECausalFS.structurefitestimator import *
from HTECausalFS.htefitestimator import *


class HTEFSEstimator:

    def __init__(self, hte_estimator, learner_econ=False, learner_nn=False, dl_layers=None, dl_params=None,
                 # GCFS params (bad way to do it like this for now i don't really care
                 gcfs_method=TauRisk, gcfs_params=None,
                 forward=True, backward=False, select_forward=False, select_backwards=False, k=10,
                 hte_selector=None, hte_selector_econ=False, hte_selector_nn=False, verbose=False,
                 num_features_to_select=np.inf, num_features_to_remove=np.inf,
                 # LCFS params
                 pc_alg="CMB", edge_alg="reci",
                 adjust_method="nonforbidden",
                 use_propensity=False, propensity_model=LogisticRegression, propensity_params=None, alpha=0.05,
                 binary_data=False, check_colliders=False):
        self.HTE_Estimator = hte_estimator
        self.dl_layers = dl_layers
        self.dl_params = dl_params
        self.temp_hte_estimator = None

        self.learner_econ = learner_econ
        self.learner_nn = learner_nn

        self.general_params = {"hte_estimator": hte_estimator, "learner_econ": learner_econ, "learner_nn": learner_nn}

        self.gcfs_params = {"gcfs_method": gcfs_method, "gcfs_params": gcfs_params, "forward": forward,
                            "backward": backward, "select_forward": select_forward,
                            "select_backwards": select_backwards, "k": k, "hte_selector": hte_selector,
                            "hte_selector_econ": hte_selector_econ,
                            "hte_selector_nn": hte_selector_nn, "num_features_to_select": num_features_to_select,
                            "num_features_to_remove": num_features_to_remove, "verbose": verbose}
        self.lcfs_params = {"pc_alg": pc_alg, "edge_alg": edge_alg, "adjust_method": adjust_method,
                            "use_propensity": use_propensity, "propensity_model": propensity_model,
                            "propensity_params": propensity_params, "alpha": alpha, "binary_data": binary_data,
                            "check_colliders": check_colliders, }

        self.lcfs_estimator = StructureFitEstimator(**self.general_params, **self.lcfs_params)
        self.gcfs_estimator = HTEFitEstimator(**self.general_params, **self.gcfs_params)

        self.adjustment_set = []

    def find_adjustment_set(self, df):
        self.gcfs_estimator.find_adjustment_set(df)
        gcfs_adjustment = list(self.gcfs_estimator.adjustment_set)

        after_gcfs_select_df = df[gcfs_adjustment + ["t", "y"]]
        self.lcfs_estimator.find_adjustment_set(after_gcfs_select_df, None)
        final_adjustment = list(self.lcfs_estimator.adjustment_set)

        self.adjustment_set = final_adjustment

    def fit(self, x, y, t):

        dat_dict = dict()
        for j in range(x.shape[1]):
            dat_dict[f"x{j}"] = x[:, j]
        dat_dict["t"] = t
        dat_dict["y"] = y
        df = pd.DataFrame(dat_dict)

        self.find_adjustment_set(df)

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
