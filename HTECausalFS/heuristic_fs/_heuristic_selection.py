# from HTECausalFS.util import *
import numpy as np
from sklearn.model_selection import train_test_split


class _HeuristicSelection:
    def __init__(self, split_size=0.5, train_all=False, train_set=False, ground_truth=False, seed=724, verbose=False):
        self.split_size = split_size
        self.seed = seed
        self.verbose = verbose

        self.eval_method = None

        self.fit = False

        self.train_all = train_all
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
                    if self.train_all:
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
                    if self.train_all:
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
            if self.train_all:
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
            if self.train_all:
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
                    if self.train_all:
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
                    if self.train_all:
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
