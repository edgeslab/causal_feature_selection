from HTECausalFS.heuristic_fs.heuristic_selection import TauRisk, PEHESelection, CounterfactualCrossValidation, \
    PlugInTau
import pandas as pd
import numpy as np


GCFS_lookup = {
    "taurisk": TauRisk,
    "tau": TauRisk,
    "tau risk": TauRisk,
    "plugin": PlugInTau,
    "plugintau": PlugInTau,
    "peheselection": PEHESelection,
    "pehe": PEHESelection,
    "counterfactualcrossvalidation": CounterfactualCrossValidation,
    "cfcv": CounterfactualCrossValidation,
    "counterfactual cross validation": CounterfactualCrossValidation,
}


class HTEFitEstimator:

    def __init__(self, hte_estimator, learner_econ=False, learner_nn=False, gcfs_method=TauRisk, gcfs_params=None,
                 forward=False, backward=False, select_forward=False, select_backwards=False, k=10,
                 rival_genetic_algorithm=False, pop_size=100, num_generations=100, fast_crossover=True,
                 competitive_swarm=False, swarm_size=100, max_generations=200,
                 hte_selector=None, hte_selector_econ=False, hte_selector_nn=False, verbose=False,
                 dl_layers=None, dl_params=None, num_features_to_select=np.inf, num_features_to_remove=np.inf):

        self.verbose = verbose

        if isinstance(gcfs_method, str):
            gcfs_method = GCFS_lookup[gcfs_method.lower()]
        self.gcfs_params = gcfs_params
        if self.gcfs_params is not None:
            self.gcfs_method = gcfs_method(**gcfs_params, verbose=verbose)
        else:
            self.gcfs_method = gcfs_method(verbose=verbose)

        self.hte_estimator = hte_estimator
        self.learner_econ = learner_econ
        self.learner_nn = learner_nn
        self.dl_layers = dl_layers
        self.dl_params = dl_params
        self.adjustment_set = None

        # ----------------------------------------------------------------
        # choosing which heuristic to use
        # ----------------------------------------------------------------
        self.forward = forward
        self.backward = backward
        self.select_forward = select_forward
        self.select_backward = select_backwards
        self.rival_genetic_algorithm = rival_genetic_algorithm
        self.competitive_swarm = competitive_swarm

        # ----------------------------------------------------------------
        # heuristic params
        # ----------------------------------------------------------------
        self.k = k  # for top-k selection
        self.g_pop_size = pop_size
        self.num_generations = num_generations
        self.fast_crossover = fast_crossover
        self.swarm_size = swarm_size
        self.max_generations = max_generations

        self.hte_selector = hte_selector
        self.hte_selector_econ = hte_selector_econ
        self.hte_selector_nn = hte_selector_nn

        self.temp_hte_estimator = None

        self.num_features_to_select = num_features_to_select
        self.num_features_to_remove = num_features_to_remove

    def find_adjustment_set(self, df):

        est = self.hte_estimator
        learner_econ = self.learner_econ
        learner_nn = self.learner_nn
        if self.hte_selector is not None:
            est = self.hte_selector
            learner_econ = self.hte_selector_econ
            learner_nn = self.hte_selector_nn

        if self.forward:
            self.adjustment_set = self.gcfs_method.forward_selection(est, df,
                                                                     metalearner=learner_econ,
                                                                     neural_network=learner_nn,
                                                                     n_features_to_select=self.num_features_to_select)
        elif self.backward:
            self.adjustment_set = self.gcfs_method.backward_selection(est, df,
                                                                      metalearner=learner_econ,
                                                                      neural_network=learner_nn,
                                                                      n_features_to_select=self.num_features_to_select,
                                                                      n_features_to_remove=self.num_features_to_remove)
        elif self.select_forward:
            self.adjustment_set = self.gcfs_method.select_top_k_forwards(est, df, metalearner=learner_econ,
                                                                         neural_network=learner_nn, k=self.k)
        elif self.select_backward:
            self.adjustment_set = self.gcfs_method.select_top_k_backwards(est, df, metalearner=learner_econ,
                                                                          neural_network=learner_nn, k=self.k)
        elif self.rival_genetic_algorithm:
            self.adjustment_set = self.gcfs_method.rival_genetic_algorithm(est, df, metalearner=learner_econ,
                                                                           neural_network=learner_nn,
                                                                           pop_size=self.g_pop_size,
                                                                           num_generations=self.num_generations,
                                                                           fast_crossover=self.fast_crossover)
        elif self.competitive_swarm:
            self.adjustment_set = self.gcfs_method.competitive_swarm(est, df, metalearner=learner_econ,
                                                                     neural_network=learner_nn,
                                                                     swarm_size=self.swarm_size,
                                                                     max_generations=self.max_generations)
        else:
            self.adjustment_set = self.gcfs_method.forward_selection(est, df,
                                                                     metalearner=learner_econ,
                                                                     neural_network=learner_nn)

    def fit(self, x, y, t):

        dat_dict = dict()
        for j in range(x.shape[1]):
            dat_dict[f"x{j}"] = x[:, j]
        dat_dict["t"] = t
        dat_dict["y"] = y
        df = pd.DataFrame(dat_dict)

        self.find_adjustment_set(df)

        new_x = df[self.adjustment_set].values
        if len(new_x.shape) < 2:
            new_x = new_x.reshape(-1, 1)

        if self.learner_econ:
            self.hte_estimator.fit(y, t, X=new_x)
        elif self.learner_nn:
            # temp_learner = learner(n_features=x_train.shape[1])
            pre_treatment, post_treatment = self.dl_layers(new_x.shape[1])
            params = self.dl_params
            self.temp_hte_estimator = self.hte_estimator(pre_treatment=pre_treatment, post_treatment=post_treatment,
                                                         **params)
            self.temp_hte_estimator.fit(new_x, y, t)
        else:
            self.hte_estimator.fit(new_x, y, t)

    def predict(self, x):

        dat_dict = dict()
        for j in range(x.shape[1]):
            dat_dict[f"x{j}"] = x[:, j]
        df = pd.DataFrame(dat_dict)

        new_x = df[self.adjustment_set].values
        if len(new_x.shape) < 2:
            new_x = new_x.reshape(-1, 1)

        if self.learner_econ:
            pred = self.hte_estimator.effect(new_x)
        elif self.learner_nn:
            pred = self.temp_hte_estimator.predict(new_x)
        else:
            pred = self.hte_estimator.predict(new_x)

        return pred