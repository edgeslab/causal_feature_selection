# from HTECausalFS.util import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from HTECausalFS.estimators.bnn import BalancingNeuralNetwork
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from scipy.spatial import cKDTree

# util
from HTECausalFS.heuristic_fs.heuristic_util import simple_selection, simple_mutation, simple_crossover


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

        if self.ground_truth or "effect" in data.columns:
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

    def forward_selection(self, estimator, data, metalearner=False, neural_network=False, nn_params=None):
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

                pred = self._fit_estimator(estimator, x, y, t, x_test, metalearner, neural_network, nn_params)

                temp_risk = self._get_risk(all_x_test, y_test, t_test, pred, all_x_train, x, y, t, x_test)
                if temp_risk < min_risk or min_keep is None:
                    min_keep = col
                    min_risk = temp_risk

            if min_keep is not None and current_risk > min_risk:
                keep_cols.append(min_keep)
                current_risk = min_risk
                other_cols = [i for i in other_cols if i not in keep_cols]
            else:
                exit_flag = True

        self.reset()
        return keep_cols

    def backward_selection(self, estimator, data, metalearner=False, neural_network=False, nn_params=None):
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

        pred = self._fit_estimator(estimator, x, y, t, x_test, metalearner, neural_network, nn_params)

        current_risk = self._get_risk(all_x_test, y_test, t_test, pred, all_x_train, x, y, t, x_test)

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
                pred = self._fit_estimator(estimator, x, y, t, x_test, metalearner, neural_network, nn_params)

                temp_risk = self._get_risk(all_x_test, y_test, t_test, pred, all_x_train, x, y, t, x_test)

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

    def select_top_k_backwards(self, estimator, data, metalearner=False, neural_network=False, k=10, nn_params=None):
        np.random.seed(self.seed)
        outcome_col = "y"
        treatment_col = "t"
        effect_col = "effect"
        ignore_cols = [outcome_col, treatment_col, effect_col]
        full_cols = [i for i in data.columns if i not in ignore_cols]
        other_cols = [i for i in data.columns if i not in ignore_cols]

        processed_data = self._process_initial_data(data)
        y, y_test = processed_data["y"], processed_data["y_test"]
        t, t_test = processed_data["t"], processed_data["t_test"]
        train_indices, test_indices = processed_data["train_indices"], processed_data["test_indices"]
        all_x_train, all_x_test = processed_data["all_x_train"], processed_data["all_x_test"]

        col_scores = {}
        for col in other_cols:
            temp_cols = [i for i in other_cols if i != col]
            if len(temp_cols) < 1:
                break
            full_x = data[temp_cols].values
            x = full_x[train_indices]
            x_test = full_x[test_indices]
            pred = self._fit_estimator(estimator, x, y, t, x_test, metalearner, neural_network, nn_params)

            temp_risk = self._get_risk(all_x_test, y_test, t_test, pred, all_x_train, x, y, t, x_test)

            col_scores[col] = temp_risk

        sorted_dict = {k: v for k, v in sorted(col_scores.items(), key=lambda item: item[1], reverse=False)}
        sorted_cols = list(sorted_dict.keys())
        keep_cols = sorted_cols[:k]

        # if we somehow didn't find variables to keep, then we default to all variables
        if len(keep_cols) < 1:
            keep_cols = full_cols

        # after training we need to reset all estimators
        self.reset()
        return keep_cols

    def select_top_k_forwards(self, estimator, data, metalearner=False, neural_network=False, k=10, nn_params=None):
        np.random.seed(self.seed)
        outcome_col = "y"
        treatment_col = "t"
        effect_col = "effect"
        ignore_cols = [outcome_col, treatment_col, effect_col]
        full_cols = [i for i in data.columns if i not in ignore_cols]
        other_cols = [i for i in data.columns if i not in ignore_cols]

        processed_data = self._process_initial_data(data)
        y, y_test = processed_data["y"], processed_data["y_test"]
        t, t_test = processed_data["t"], processed_data["t_test"]
        train_indices, test_indices = processed_data["train_indices"], processed_data["test_indices"]
        all_x_train, all_x_test = processed_data["all_x_train"], processed_data["all_x_test"]

        col_scores = {}
        for col in other_cols:
            temp_cols = [col]
            if len(temp_cols) < 1:
                break
            full_x = data[temp_cols].values
            x = full_x[train_indices]
            x_test = full_x[test_indices]

            pred = self._fit_estimator(estimator, x, y, t, x_test, metalearner, neural_network, nn_params)

            temp_risk = self._get_risk(all_x_test, y_test, t_test, pred, all_x_train, x, y, t, x_test)

            col_scores[col] = temp_risk

        sorted_dict = {k: v for k, v in sorted(col_scores.items(), key=lambda item: item[1], reverse=False)}
        sorted_cols = list(sorted_dict.keys())
        keep_cols = sorted_cols[:k]

        # if we somehow didn't find variables to keep, then we default to all variables
        if len(keep_cols) < 1:
            keep_cols = full_cols

        # after training we need to reset all estimators
        self.reset()
        return keep_cols

    def simple_genetic_algorithm(self, estimator, data, pop_size=100, num_generations=10, metalearner=False,
                                 neural_network=False, nn_params=None, crossover_rate=None, mutation_rate=None):
        outcome_col = "y"
        treatment_col = "t"
        effect_col = "effect"
        ignore_cols = [outcome_col, treatment_col, effect_col]
        full_cols = [i for i in data.columns if i not in ignore_cols]
        other_cols = [i for i in data.columns if i not in ignore_cols]

        processed_data = self._process_initial_data(data)
        y, y_test = processed_data["y"], processed_data["y_test"]
        t, t_test = processed_data["t"], processed_data["t_test"]
        train_indices, test_indices = processed_data["train_indices"], processed_data["test_indices"]
        all_x_train, all_x_test = processed_data["all_x_train"], processed_data["all_x_test"]

        rng = np.random.default_rng(seed=self.seed)
        chromosomes = rng.choice([0, 1], size=(pop_size, all_x_train.shape[1]))
        new_chromosomes = np.zeros(chromosomes.shape)

        self.reset()
        for i in range(num_generations):
            scores = np.zeros(chromosomes.shape[0])
            for idx, chrom in enumerate(chromosomes):
                x = all_x_train[:, chrom == 1]
                x_test = all_x_test[:, chrom == 1]
                pred = self._fit_estimator(estimator, x, y, t, x_test, metalearner, neural_network, nn_params)

                temp_risk = self._get_risk(all_x_test, y_test, t_test, pred, all_x_train, x, y, t, x_test)

                scores[idx] = temp_risk

            scores = (1 - scores) / np.sum(1 - scores)

            for j in range(pop_size):
                chrom1, chrom2 = simple_selection(scores, chromosomes, rng=rng)

                new_chrom = simple_crossover(chrom1, chrom2, crossover_rate=crossover_rate, rng=rng)

                new_chrom = simple_mutation(new_chrom, mutation_rate=mutation_rate, rng=rng)

                new_chromosomes[j] = new_chrom

            chromosomes = new_chromosomes.copy()

        scores = np.zeros(chromosomes.shape[0])
        for idx, chrom in enumerate(chromosomes):
            x = all_x_train[:, chrom == 1]
            x_test = all_x_test[:, chrom == 1]
            pred = self._fit_estimator(estimator, x, y, t, x_test, metalearner, neural_network, nn_params)

            temp_risk = self._get_risk(all_x_test, y_test, t_test, pred, all_x_train, x, y, t, x_test)

            scores[idx] = temp_risk

        sorted_idx = np.argsort(scores)
        selected_chrom = chromosomes[sorted_idx[0]]
        keep_cols = [col for i, col in enumerate(full_cols) if selected_chrom[i] == 1]

        return keep_cols

    def rival_genetic_algorithm(self, estimator, data, pop_size=100, num_generations=10, metalearner=False,
                                neural_network=False, nn_params=None, fast_crossover=False):
        outcome_col = "y"
        treatment_col = "t"
        effect_col = "effect"
        ignore_cols = [outcome_col, treatment_col, effect_col]
        full_cols = [i for i in data.columns if i not in ignore_cols]
        other_cols = [i for i in data.columns if i not in ignore_cols]

        processed_data = self._process_initial_data(data)
        y, y_test = processed_data["y"], processed_data["y_test"]
        t, t_test = processed_data["t"], processed_data["t_test"]
        train_indices, test_indices = processed_data["train_indices"], processed_data["test_indices"]
        all_x_train, all_x_test = processed_data["all_x_train"], processed_data["all_x_test"]

        rng = np.random.default_rng(seed=self.seed)
        feature_size = all_x_train.shape[1]
        chromosomes = rng.choice([0, 1], size=(pop_size, feature_size))
        chrom_idx = np.arange(pop_size)

        best_score = np.inf
        best_chrom = np.zeros(feature_size)

        if pop_size % 2 == 1:
            pop_size = pop_size + 1  # guarantee even population for dividing by two

        H = dict()  # for keeping an archive

        self.reset()
        for g in range(num_generations):
            scores = np.zeros(chromosomes.shape[0])
            if fast_crossover:
                Tu = num_generations / 2
                if g < Tu:
                    num_crossovers = int(np.round((pop_size - 1) * (1 - g / Tu)))
                else:
                    num_crossovers = int(1 + np.round((pop_size - 1) * ((g - Tu) / Tu)))
            else:
                num_crossovers = pop_size
            for idx, chrom in enumerate(chromosomes):

                chrom_str = str(chrom)

                if chrom_str not in H:
                    x = all_x_train[:, chrom == 1]
                    x_test = all_x_test[:, chrom == 1]
                    pred = self._fit_estimator(estimator, x, y, t, x_test, metalearner, neural_network, nn_params)

                    temp_risk = self._get_risk(all_x_test, y_test, t_test, pred, all_x_train, x, y, t, x_test)

                    scores[idx] = temp_risk

                    H[chrom_str] = temp_risk
                else:
                    scores[idx] = H[chrom_str]

            # favg = np.mean(risks/np.sum(risks))
            # favg = np.mean(risks)
            favg = 1 - np.mean((scores - np.min(scores)) / (np.max(scores) - np.min(scores)))
            mutation_rate = favg - g * (favg / num_generations)

            # choose two
            rng.shuffle(chrom_idx)
            pairs = [(chrom_idx[i], chrom_idx[i + 1]) for i in range(0, pop_size, 2)]

            # competitions
            winners = []
            losers = []
            for pair in pairs:
                first_score = scores[pair[0]]
                second_score = scores[pair[1]]
                if first_score <= second_score:
                    winners.append(pair[0])
                    losers.append(pair[1])
                else:
                    winners.append(pair[1])
                    losers.append(pair[0])

            # selection and crossover
            winner_scores = scores[winners]
            loser_scores = scores[losers]

            winner_probs = (1 - winner_scores) / (np.sum(1 - winner_scores))
            loser_probs = (1 - loser_scores) / (np.sum(1 - loser_scores))

            new_chromosomes = np.zeros((num_crossovers, feature_size))
            for j in range(num_crossovers):
                # select two parents from winner and one parent from losers
                winner_parent_idx = rng.choice(winners, p=winner_probs, size=2, replace=False)
                loser_parent_idx = rng.choice(losers, p=loser_probs, size=1, replace=False)

                # crossover
                winner_chrom1 = chromosomes[winner_parent_idx[0]]
                winner_chrom2 = chromosomes[winner_parent_idx[1]]
                loser_chrom = chromosomes[loser_parent_idx[0]]
                new_chrom = np.zeros(chromosomes.shape[1])
                stochastic_crossover_vector = rng.random(size=feature_size)
                new_chrom[stochastic_crossover_vector > 2 / 3] = loser_chrom[stochastic_crossover_vector > 2 / 3]
                new_chrom[stochastic_crossover_vector <= 2 / 3] = winner_chrom2[stochastic_crossover_vector <= 2 / 3]
                new_chrom[stochastic_crossover_vector <= 1 / 3] = winner_chrom1[stochastic_crossover_vector <= 1 / 3]

                # mutation
                mutation_random_vector = rng.random(size=feature_size)
                new_chrom[mutation_rate >= mutation_random_vector] = 1 - new_chrom[
                    mutation_rate >= mutation_random_vector]

                new_chromosomes[j] = new_chrom

            new_scores = np.zeros(num_crossovers)
            for idx, chrom in enumerate(new_chromosomes):

                chrom_str = str(chrom)

                if chrom_str not in H:
                    x = all_x_train[:, chrom == 1]
                    x_test = all_x_test[:, chrom == 1]
                    # print(g, x.shape, x_test.shape)
                    pred = self._fit_estimator(estimator, x, y, t, x_test, metalearner, neural_network, nn_params)

                    temp_risk = self._get_risk(all_x_test, y_test, t_test, pred, all_x_train, x, y, t, x_test)

                    new_scores[idx] = temp_risk

                    H[chrom_str] = temp_risk
                else:
                    new_scores[idx] = H[chrom_str]

            all_chroms = np.vstack((chromosomes, new_chromosomes))
            all_risks = np.concatenate((scores, new_scores))

            sorted_all = np.argsort(all_risks)
            chromosomes = all_chroms[sorted_all[:pop_size]].copy()
            gen_score = all_risks[sorted_all[0]]
            if all_risks[sorted_all[0]] < best_score:
                best_score = gen_score
                best_chrom = chromosomes[0]

            # print(len(H))

        keep_cols = [col for i, col in enumerate(full_cols) if best_chrom[i] == 1]

        return keep_cols

    def competitive_swarm(self, estimator, data, swarm_size=100, max_generations=200, metalearner=False,
                          neural_network=False, nn_params=None, threshold=0.5, phi=0.1):
        outcome_col = "y"
        treatment_col = "t"
        effect_col = "effect"
        ignore_cols = [outcome_col, treatment_col, effect_col]
        full_cols = [i for i in data.columns if i not in ignore_cols]
        other_cols = [i for i in data.columns if i not in ignore_cols]

        processed_data = self._process_initial_data(data)
        y, y_test = processed_data["y"], processed_data["y_test"]
        t, t_test = processed_data["t"], processed_data["t_test"]
        train_indices, test_indices = processed_data["train_indices"], processed_data["test_indices"]
        all_x_train, all_x_test = processed_data["all_x_train"], processed_data["all_x_test"]

        time = 0
        H = dict()

        feature_size = all_x_train.shape[1]

        rng = np.random.default_rng(self.seed)

        particle_positions = rng.random(size=(swarm_size, feature_size))
        particle_velocities = np.zeros((swarm_size, feature_size))
        particle_idx = np.arange(swarm_size)

        best_error = np.inf
        best_particle = np.ones(feature_size)

        # while loop
        change_flag = True
        while change_flag and time < max_generations:
            r1 = rng.random(size=feature_size)
            r2 = rng.random(size=feature_size)
            r3 = rng.random(size=feature_size)

            change_flag = False
            for p in range(swarm_size):
                use_idx = particle_positions[p] > threshold
                use_idx_str = str(use_idx)
                if use_idx_str not in H:
                    # calculate fitness score
                    x = all_x_train[:, use_idx]
                    x_test = all_x_test[:, use_idx]
                    pred = self._fit_estimator(estimator, x, y, t, x_test, metalearner, neural_network, nn_params)

                    temp_risk = self._get_risk(all_x_test, y_test, t_test, pred, all_x_train, x, y, t, x_test)
                    H[use_idx_str] = temp_risk

                    if temp_risk < best_error:
                        best_particle = particle_positions[p]
                        best_error = temp_risk
                    change_flag = True

            # choose two
            rng.shuffle(particle_idx)
            pairs = [(particle_idx[i], particle_idx[i + 1])
                     for i in range(0, swarm_size, 2)]

            curr_particle_mean = np.mean(particle_positions, axis=0)
            for pair in pairs:
                pair1_str = str(particle_positions[pair[0]] > threshold)
                pair2_str = str(particle_positions[pair[1]] > threshold)

                competition_check = H[pair1_str] > H[pair2_str]
                winning_particle = pair[0] if competition_check else pair[1]
                losing_particle = pair[1] if competition_check else pair[0]

                vel_term1 = r1 * particle_velocities[losing_particle]
                vel_term2 = r2 * (particle_positions[winning_particle] - particle_positions[losing_particle])
                vel_term3 = phi * r3 * (curr_particle_mean - particle_positions[losing_particle])
                particle_velocities[losing_particle] = vel_term1 + vel_term2 + vel_term3

                particle_positions[losing_particle] += particle_velocities[losing_particle]

            time += 1

        keep_cols = [col for i, col in enumerate(full_cols) if best_particle[i] > threshold]

        return keep_cols

    def _fit_estimator(self, estimator, x, y, t, x_test, metalearner, neural_network, nn_params=None):
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
            if nn_params is not None:
                temp_est = estimator(**nn_params)
            else:
                temp_est = estimator(x.shape[1])
            temp_est.fit(x, y, t)
            pred = temp_est.predict(x_test)
        # ----------------------------------------------------------------
        # Causal trees use x, y, t unlike ECONML
        # ----------------------------------------------------------------
        else:
            estimator.fit(x, y, t)
            pred = estimator.predict(x_test)

        return pred

    def _get_risk(self, all_x_test, y_test, t_test, pred, all_x_train, x, y, t, x_test):
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

        return temp_risk


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
