# from HTECausalFS.util import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from HTECausalFS.estimators.bnn import BalancingNeuralNetwork
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from scipy.spatial import cKDTree


class _HeuristicSelection:
    def __init__(self, split_size=0.5, train_all_features=True, train_set=False, ground_truth=False, seed=724,
                 verbose=False):
        self.split_size = split_size
        self.seed = seed
        self.verbose = verbose

        self.eval_method = None

        self.fit = False

        self.train_all_features = train_all_features
        self.train_set = train_set
        self.ground_truth = ground_truth

    def reset(self):
        self.fit = False

    # ----------------------------------------------------------------
    # Get various views of the data
    # train = data to train estimators on
    # test = "validation set" e.g. for tau risk and training intermediate estimators
    # ----------------------------------------------------------------
    def _process_initial_data(self, data):
        np.random.seed(self.seed)
        outcome_col = "y"
        treatment_col = "t"

        ignore_cols = [outcome_col, treatment_col]

        if self.ground_truth:
            effect_col = "effect"
            ignore_cols = ignore_cols + [effect_col]

        other_cols = [i for i in data.columns if i not in ignore_cols]

        all_x = data[other_cols].values
        full_x = data[other_cols].values
        full_y = data[outcome_col].values
        full_t = data[treatment_col].values

        indices = np.arange(full_y.shape[0])
        train_indices, test_indices = train_test_split(indices, test_size=self.split_size, random_state=self.seed)
        x = full_x[train_indices]
        x_test = full_x[test_indices]
        y = full_y[train_indices]
        y_test = full_y[test_indices]
        t = full_t[train_indices]
        t_test = full_t[test_indices]

        all_x_train = all_x[train_indices]
        all_x_test = all_x[test_indices]

        return {
            "x": x, "x_test": x_test, "full_x": full_x, "all_x_train": all_x_train, "all_x_test": all_x_test, "y": y,
            "y_test": y_test, "t": t, "t_test": t_test, "train_indices": train_indices, "test_indices": test_indices
        }

    def forward_selection(self, estimator, data, metalearner=False, neural_network=False):
        np.random.seed(self.seed)
        outcome_col = "y"
        treatment_col = "t"
        effect_col = "effect"
        ignore_cols = [outcome_col, treatment_col, effect_col]
        other_cols = [i for i in data.columns if i not in ignore_cols]

        processed_data = self._process_initial_data(data)
        y, y_test = processed_data["y"], processed_data["y_test"]
        t, t_test = processed_data["t"], processed_data["t_test"]
        train_indices, test_indices = processed_data["train_indices"], processed_data["test_indices"]
        all_x_train, all_x_test = processed_data["all_x_train"], processed_data["all_x_test"]

        current_risk = np.inf

        exit_flag = False
        keep_cols = []
        while not exit_flag:
            if self.verbose:
                print(current_risk, keep_cols)
            min_risk = np.inf
            min_keep = None
            for col in other_cols:
                temp_cols = keep_cols + [col]
                if len(temp_cols) < 1:
                    break
                full_x = data[temp_cols].values
                x = full_x[train_indices]
                x_test = full_x[test_indices]
                # ----------------------------------------------------------------
                # check which estimator we are using (ECONML uses different format)
                # ----------------------------------------------------------------
                if metalearner:
                    estimator.fit(y, t, X=x)
                    pred = estimator.effect(x_test)
                # ----------------------------------------------------------------
                # Use a temp estimator for NNs to prevent potential problems from
                # the way I implemented it...
                # ----------------------------------------------------------------
                elif neural_network:
                    temp_est = estimator(x.shape[1])
                    temp_est.fit(x, y, t)
                    pred = temp_est.predict(x_test)
                # ----------------------------------------------------------------
                # Causal trees use x, y, t unlike ECONML
                # ----------------------------------------------------------------
                else:
                    estimator.fit(x, y, t)
                    pred = estimator.predict(x_test)

                # ----------------------------------------------------------------
                # Train the evaluation metric + get evaluation
                # ----------------------------------------------------------------
                if self.train_set:
                    # Train on all features and the training set
                    if self.train_all_features:
                        temp_risk = self.eval_method(all_x_test, y_test, t_test, pred, train_x=all_x_train, train_y=y,
                                                     train_t=t)
                    # Train on subset features and the training set
                    else:
                        self.reset()
                        temp_risk = self.eval_method(x_test, y_test, t_test, pred, train_x=x, train_y=y, train_t=t)
                else:
                    # ----------------------------------------------------------------
                    # My original default case (all features + test set)
                    # ----------------------------------------------------------------
                    if self.train_all_features:
                        temp_risk = self.eval_method(all_x_test, y_test, t_test, pred)
                    # Train on subset features + test set
                    else:
                        self.reset()
                        temp_risk = self.eval_method(x_test, y_test, t_test, pred)
                if temp_risk < min_risk or min_keep is None:
                    min_keep = col
                    min_risk = temp_risk

            if min_keep is not None and current_risk > min_risk:
                keep_cols.append(min_keep)
                current_risk = min_risk
                other_cols = [i for i in other_cols if i not in keep_cols]
            else:
                exit_flag = True

        # self.reset()
        return keep_cols

    def backward_selection(self, estimator, data, metalearner=False, neural_network=False):
        np.random.seed(self.seed)
        outcome_col = "y"
        treatment_col = "t"
        effect_col = "effect"
        ignore_cols = [outcome_col, treatment_col, effect_col]
        full_cols = [i for i in data.columns if i not in ignore_cols]
        other_cols = [i for i in data.columns if i not in ignore_cols]

        processed_data = self._process_initial_data(data)
        x, x_test = processed_data["x"], processed_data["x_test"]
        y, y_test = processed_data["y"], processed_data["y_test"]
        t, t_test = processed_data["t"], processed_data["t_test"]
        train_indices, test_indices = processed_data["train_indices"], processed_data["test_indices"]
        all_x_train, all_x_test = processed_data["all_x_train"], processed_data["all_x_test"]

        # ----------------------------------------------------------------
        # check which estimator we are using (ECONML uses different format)
        # ----------------------------------------------------------------
        if metalearner:
            estimator.fit(y, t, X=x)
            pred = estimator.effect(x_test)
        # ----------------------------------------------------------------
        # Use a temp estimator for NNs to prevent potential problems from
        # the way I implemented it...
        # ----------------------------------------------------------------
        elif neural_network:
            temp_est = estimator(x.shape[1])
            temp_est.fit(x, y, t)
            pred = temp_est.predict(x_test)
        # ----------------------------------------------------------------
        # Causal trees use x, y, t unlike ECONML
        # ----------------------------------------------------------------
        else:
            estimator.fit(x, y, t)
            pred = estimator.predict(x_test)

        # ----------------------------------------------------------------
        # Fixed current risk for backward...
        # ----------------------------------------------------------------
        # current_risk = tau_risk(all_x_train, y, t, all_x_test, y_test, t_test, pred)
        if self.train_set:
            # Train on all features and the training set
            if self.train_all_features:
                current_risk = self.eval_method(all_x_test, y_test, t_test, pred, train_x=all_x_train,
                                                train_y=y,
                                                train_t=t)
            # Train on subset features and the training set
            else:
                self.reset()
                current_risk = self.eval_method(x_test, y_test, t_test, pred, train_x=x, train_y=y, train_t=t)
        else:
            # ----------------------------------------------------------------
            # My original default case (all features + test set)
            # ----------------------------------------------------------------
            if self.train_all_features:
                current_risk = self.eval_method(all_x_test, y_test, t_test, pred)
            # Train on subset features + test set
            else:
                self.reset()
                current_risk = self.eval_method(x_test, y_test, t_test, pred)

        exit_flag = False
        remove_cols = []
        while not exit_flag:
            if self.verbose:
                print(current_risk, remove_cols)
            min_risk = np.inf
            min_risk_rm = None
            for col in other_cols:
                temp_cols = [i for i in other_cols if i != col]
                if len(temp_cols) < 1:
                    break
                full_x = data[temp_cols].values
                x = full_x[train_indices]
                x_test = full_x[test_indices]
                # ----------------------------------------------------------------
                # check which estimator we are using (ECONML uses different format)
                # ----------------------------------------------------------------
                if metalearner:
                    estimator.fit(y, t, X=x)
                    pred = estimator.effect(x_test)
                # ----------------------------------------------------------------
                # Use a temp estimator for NNs to prevent potential problems from
                # the way I implemented it...
                # ----------------------------------------------------------------
                elif neural_network:
                    temp_est = estimator(x.shape[1])
                    temp_est.fit(x, y, t)
                    pred = temp_est.predict(x_test)
                # ----------------------------------------------------------------
                # Causal trees use x, y, t unlike ECONML
                # ----------------------------------------------------------------
                else:
                    estimator.fit(x, y, t)
                    pred = estimator.predict(x_test)

                # ----------------------------------------------------------------
                # Train the evaluation metric + get evaluation
                # ----------------------------------------------------------------
                if self.train_set:
                    # Train on all features and the training set
                    if self.train_all_features:
                        temp_risk = self.eval_method(all_x_test, y_test, t_test, pred, train_x=all_x_train,
                                                     train_y=y,
                                                     train_t=t)
                    # Train on subset features and the training set
                    else:
                        self.reset()
                        temp_risk = self.eval_method(x_test, y_test, t_test, pred, train_x=x, train_y=y, train_t=t)
                else:
                    # ----------------------------------------------------------------
                    # My original default case (all features + test set)
                    # ----------------------------------------------------------------
                    if self.train_all_features:
                        temp_risk = self.eval_method(all_x_test, y_test, t_test, pred)
                    # Train on subset features + test set
                    else:
                        self.reset()
                        temp_risk = self.eval_method(x_test, y_test, t_test, pred)
                if temp_risk < min_risk:
                    min_risk_rm = col
                    min_risk = temp_risk

            if min_risk_rm is not None and current_risk > min_risk:
                remove_cols.append(min_risk_rm)
                current_risk = min_risk
                other_cols = [i for i in other_cols if i not in remove_cols]
            else:
                exit_flag = True

        keep_cols = [i for i in full_cols if i not in remove_cols]

        # if we somehow didn't find variables to keep, then we default to all variables
        if len(keep_cols) < 1:
            keep_cols = full_cols

        # after training we need to reset all estimators
        self.reset()
        return keep_cols


class CounterfactualCrossValidation(_HeuristicSelection):
    def __init__(self, epochs=100, **kwargs):
        super().__init__(**kwargs)

        self.epochs = epochs
        self.propensity_method = LogisticRegression(max_iter=100000, solver="saga")
        self.pred_method = BalancingNeuralNetwork

        self.prediction_method = None

        self.eval_method = self.eval

    def eval(self, x, y, t, pred, train_x=None, train_y=None, train_t=None):
        if train_x is None or train_y is None or train_t is None:
            train_x = x
            train_y = y
            train_t = t

        if not self.fit:
            self.propensity_method.fit(train_x, train_t)
            self.prediction_method = self.pred_method(x.shape[1])
            self.prediction_method.fit(train_x, train_y, train_t)
            self.fit = True

        propensity = self.propensity_method.predict_proba(x)[:, -1]
        ft1, ft0 = self.prediction_method.forward_numpy(x, t)
        # how to reorder them back to t==1 t==0
        treated_idx = np.where(t == 1)[0]
        control_idx = np.where(t == 0)[0]
        ft = np.zeros(x.shape[0])
        ft[treated_idx] = ft1
        ft[control_idx] = ft0

        f1_input = np.hstack((x, np.ones((x.shape[0], 1))))
        f0_input = np.hstack((x, np.zeros((x.shape[0], 1))))
        f10_input = np.vstack((f1_input, f0_input))
        f1, f0 = self.prediction_method.forward_numpy(f10_input[:, :-1], f10_input[:, -1])
        # print(f10)
        # f1 = f10[f10_input[:, -1] == 1]
        # f0 = f10[f10_input[:, -1] == 0]
        # f1 = f10_output[0]
        # f0 = f10_output[1]

        treatment_part = (t - propensity) / (propensity * (1 - propensity))
        outcome_part = (y - ft)
        doubly_part = f1 - f0

        cfcv_plugin = treatment_part * outcome_part + doubly_part

        err = np.mean((cfcv_plugin - pred) ** 2)

        return err


class PlugInTau(_HeuristicSelection):
    def __init__(self, epochs=100, **kwargs):
        super().__init__(**kwargs)

        self.epochs = epochs
        self.eval_method = self.cfr_wass

        self.cfr = None

    def cfr_wass(self, x, y, t, pred, train_x=None, train_y=None, train_t=None):
        if train_x is None or train_y is None or train_t is None:
            train_x = x
            train_y = y
            train_t = t

        if not self.fit:
            self.cfr = BalancingNeuralNetwork(n_features=x.shape[1])
            self.cfr.fit(train_x, train_y, train_t, epochs=self.epochs)
            self.fit = True

        cfr_pred = self.cfr.predict(x)
        # cfr_pred = cfr_pred.detach().numpy().reshape(-1)
        err = np.mean((cfr_pred - pred) ** 2)

        return err


class PEHESelection(_HeuristicSelection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.eval_method = self.pehe

        self.kdtree = None

    def pehe(self, x, y, t, pred, train_x=None, train_y=None, train_t=None):
        nn_effect = self._compute_nn_effect(x, y, t, train_x=train_x, train_y=train_y, train_t=train_t)

        err = np.mean((nn_effect - pred) ** 2)

        return err

    def _compute_nn_effect(self, x, y, t, k=5, train_x=None, train_y=None, train_t=None):
        if train_x is None or train_y is None or train_t is None:
            train_x = x
            train_y = y
            train_t = t
            flag = False
        else:
            flag = True
        self.kdtree = cKDTree(train_x)
        d, idx = self.kdtree.query(x, k=train_x.shape[0])
        if flag:
            idx = idx[:, 1:]
        else:
            idx = idx[:, :-1]
        treated = np.where(train_t == 1)[0]
        control = np.where(train_t == 0)[0]
        bool_treated = np.isin(idx, treated)
        bool_control = np.isin(idx, control)

        nn_effect = np.zeros(x.shape[0])
        for i in range(len(bool_treated)):
            i_treat_idx = np.where(bool_treated[i, :])[0][:k]
            i_control_idx = np.where(bool_control[i, :])[0][:k]

            i_treat_nn = train_y[idx[i, i_treat_idx]]
            i_cont_nn = train_y[idx[i, i_control_idx]]

            nn_effect[i] = np.mean(i_treat_nn) - np.mean(i_cont_nn)

        return nn_effect


class TauRisk(_HeuristicSelection):

    def __init__(self, outcome_est=GradientBoostingRegressor(), propensity_est=GradientBoostingClassifier(), **kwargs):
        super().__init__(**kwargs)

        self.eval_method = self.tau_risk
        self.outcome_est = outcome_est
        self.propensity_est = propensity_est

    def tau_risk(self, x, y, t, pred, train_x=None, train_y=None, train_t=None):
        if train_x is None or train_y is None or train_t is None:
            train_x = x
            train_y = y
            train_t = t

        # print(self.outcome_est, self.propensity_est)

        # train on "train data"
        if not self.fit:
            self.outcome_est.fit(train_x, train_y)

        # predict on "test data"
        m_est = self.outcome_est.predict(x)

        if not self.fit:
            self.propensity_est.fit(train_x, train_t)
            self.fit = True
        p_est = self.propensity_est.predict_proba(x)[:, 1]

        t_risk = (y - m_est) - (t - p_est) * pred
        t_risk = np.mean(t_risk ** 2)

        return t_risk
