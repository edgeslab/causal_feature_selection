import numpy as np

from src.util import *
import networkx as nx
import pandas as pd
from time import time
import pydot
import random


# TODO: When chain cannot be found default to no mediation -> also take this into account in experiments

class GraphGeneration:

    def __init__(self, num_variables=10, edge_prob=0.33, mediator=True, confounder=True, mediator_chain_length=0,
                 seed=724,
                 time_debug=False):
        if num_variables < 3:
            num_variables = 3

        self.mediator = mediator
        self.mediator_chain_length = mediator_chain_length

        self.confounder = confounder

        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        self.num_variables = num_variables
        self.edge_prob = edge_prob

        self.time_debug = time_debug

        self.t_position = 0
        self.y_position = num_variables - 1

        self.adjacency = None
        self.graph = None
        self.relabels = dict()
        self.relabels_reverse = dict()
        self.paths_from_t_to_y = None

        self.contains_mediation = False
        self.contains_confounding = False

        self.discard = False

        self.desc_t = None
        self.anc_y = None

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        random.seed(self.seed)

    def generate_graph(self):
        self.set_seed(self.seed)

        # ---------------------------------------
        # first generate adjacency matrix
        # ---------------------------------------
        self._generate_adjacency()

        # ---------------------------------------------------------
        # Generating graph
        # ---------------------------------------------------------
        self._generate_graph()

        # ----------------------------------------------------------------
        # initialize the "positions" of treatment and outcome
        # where treatment always precedes outcome
        #
        # 2021-03-19 - change this to sampling T position, and then random walk to "Y"
        # Start with pairs of variables with directed edge (set in _generate_graph)
        # ----------------------------------------------------------------
        # self.t_position = self.rng.integers(0, self.num_variables - 1)
        # self.y_position = self.rng.integers(self.t_position + 1, self.num_variables)

        self.desc_t = nx.algorithms.descendants(self.graph, "T")
        self.anc_y = nx.algorithms.ancestors(self.graph, "Y")

    # ----------------------------------------------------------------
    # could pick "T" then random walk.
    # for now just pick pairs
    # ----------------------------------------------------------------
    def _set_treat_outcome_nodes(self):
        candidate_paths = self._find_candidate_n_hop_paths()
        random_edge_idx = self.rng.integers(0, len(candidate_paths))

        random_edge = candidate_paths[random_edge_idx]
        self.t_position = random_edge[0]
        self.y_position = random_edge[1]

    def _find_candidate_n_hop_paths(self):
        check_1 = not self.mediator or self.mediator_chain_length == 0
        check_2 = self.confounder is None
        if check_1 and check_2:
            return list(self.graph.edges)

        # ----------------------------------------------------------------
        # Making sure the graph has parameters
        # mediator, confounder
        # ----------------------------------------------------------------
        while_iterations = 0
        while while_iterations < 100:

            while_iterations += 1

            all_n_hop_paths = np.linalg.matrix_power(self.adjacency, self.mediator_chain_length + 1)
            if self.mediator and self.mediator_chain_length > 0:
                up_to_n_adj_sum = self.adjacency.copy()
                for i in range(2, self.mediator_chain_length + 1):
                    up_to_n_adj_sum += np.linalg.matrix_power(self.adjacency, i)

                removed_below_n_hop_paths = ((up_to_n_adj_sum == 0) & (all_n_hop_paths > 0)).astype(int)

                nhop_query = removed_below_n_hop_paths == 1

                candidate_t = np.where(nhop_query)[0]
                candidate_y = np.where(nhop_query)[1]
            else:
                nhop_query = all_n_hop_paths == 1
                candidate_t = np.where(nhop_query)[0]
                candidate_y = np.where(nhop_query)[1]

            if len(candidate_t) < 1 or len(candidate_y) < 1:
                self._generate_adjacency()
                self.graph = nx.convert_matrix.from_numpy_matrix(self.adjacency, create_using=nx.DiGraph)
            else:
                # sample within this loop maybe instead of checking all paths then sampling?
                candidate_paths = [(i, j) for i, j in zip(candidate_t, candidate_y)]
                random.shuffle(candidate_paths)
                if self.confounder is not None:
                    if self.confounder:
                        confounder_candidate_paths = []
                        for path in candidate_paths:
                            if self._check_backdoor_exists(path[0], path[1]):
                                confounder_candidate_paths.append(path)
                            if len(confounder_candidate_paths) > 0:
                                return confounder_candidate_paths
                    else:
                        no_confounder_candidate_paths = []
                        for path in candidate_paths:
                            if not self._check_backdoor_exists(path[0], path[1]):
                                no_confounder_candidate_paths.append(path)
                            if len(no_confounder_candidate_paths) > 0:
                                return no_confounder_candidate_paths
                else:
                    return candidate_paths

        self.mediator = False
        self.discard = True
        return list(self.graph.edges)

    def _generate_adjacency(self):
        start = time()

        self.adjacency = np.zeros((self.num_variables, self.num_variables))
        while np.isclose(np.sum(self.adjacency), 0):
            for i in range(self.num_variables):
                for j in range(i + 1, self.num_variables):
                    create_edge_bool = self.rng.choice([True, False],
                                                       p=[self.edge_prob, 1 - self.edge_prob])
                    if create_edge_bool:
                        self.adjacency[i, j] = 1

    def _check_backdoor_exists(self, nodes1, nodes2):
        start = time()

        if len(nx.ancestors(self.graph, nodes1)) < 1:
            return False

        nodes1 = {nodes1}
        nodes2 = {nodes2}

        paths = []
        undirected_graph = self.graph.to_undirected()
        nodes12 = nodes1.union(nodes2)
        for node1 in nodes1:
            for node2 in nodes2:
                backdoor_paths = []
                all_paths = nx.all_simple_paths(undirected_graph, source=node1, target=node2)
                for pth in all_paths:
                    if self.graph.has_edge(pth[1], pth[0]):
                        backdoor_paths.append(pth)
                    if len(backdoor_paths) > 100:
                        break
                # backdoor_paths = [
                #     pth
                #     for pth in nx.all_simple_paths(undirected_graph, source=node1, target=node2)
                #     if self.graph.has_edge(pth[1], pth[0])]
                filtered_backdoor_paths = [
                    pth
                    for pth in backdoor_paths
                    if len(nodes12.intersection(pth[1:-1])) == 0]
                paths.extend(filtered_backdoor_paths)

        end = time()
        if self.time_debug:
            print(f"Check backdoor paths elapsed time: {end - start}")

        return len(paths) > 1

    def _generate_treatment_and_outcome(self):

        self.graph = nx.convert_matrix.from_numpy_matrix(self.adjacency, create_using=nx.DiGraph)

        # ----------------------------------------------------------------
        # Set position of treatment and outcome
        # ----------------------------------------------------------------
        self._set_treat_outcome_nodes()

    def _generate_graph(self):
        start = time()

        i = 0

        self._generate_treatment_and_outcome()
        # ----------------------------------------------------------------
        # Relabel nodes
        # ----------------------------------------------------------------
        self.relabels = dict()
        for node in self.graph.nodes():
            if node == self.t_position:
                self.relabels[node] = "T"
            elif node == self.y_position:
                self.relabels[node] = "Y"
            else:
                self.relabels[node] = f"X{i}"
                i += 1
        self.relabels_reverse = {y: x for x, y in self.relabels.items()}
        self.graph = nx.relabel.relabel_nodes(self.graph, mapping=self.relabels)

        # 2021-03-19: removed this in favor of sampling already set path
        # # relabel nodes and label T and Y specifically...
        # # after, force a path from T -> Y somehow... use direct edge?
        # # For now, just add a direct edge from T to Y if no path exists
        # self.all_paths_from_t_to_y = list(nx.all_simple_paths(self.graph, "T", "Y"))
        # if len(self.all_paths_from_t_to_y) < 1:
        #     self.graph.add_edge("T", "Y")

        # checking whether there is mediation or confounding
        # self.paths_from_t_to_y = list(nx.all_simple_paths(self.graph, "T", "Y"))
        # if len(self.paths_from_t_to_y) > 1:
        #     self.contains_mediation = True
        if self.mediator_chain_length > 0:
            self.contains_mediation = True
        # if self._check_backdoor_exists("T", "Y"):
        #     self.contains_confounding = True
        if self.confounder:
            self.contains_confounding = True

        end = time()
        if self.time_debug:
            print(f"Generate graph elapsed time: {end - start}")

    def show_graph(self):
        A = nx.nx_agraph.to_agraph(self.graph)
        s = A.string()
        graphs = pydot.graph_from_dot_data(s)
        svg_str = graphs[0].create_svg()
        return svg_str


class DataGeneration:
    def __init__(self, graph_generator: GraphGeneration, num_examples=1000, noise_scale=0.1, noise_strength=0.1,
                 num_hte_parents=1, hte_from_mediators=False,
                 seed=724, time_debug=False, simple_nonlinear=False, normalize_coeffs=True):
        self.graph_generator = graph_generator

        self.discard = self.graph_generator.discard

        self.num_variables = self.graph_generator.num_variables
        self.graph = self.graph_generator.graph
        self.relabels = self.graph_generator.relabels
        self.relabels_reverse = self.graph_generator.relabels_reverse
        self.t_position = self.graph_generator.t_position
        self.y_position = self.graph_generator.y_position
        self.paths_from_t_to_y = self.graph_generator.paths_from_t_to_y
        self.desc_t = self.graph_generator.desc_t
        self.anc_y = self.graph_generator.anc_y

        self.seed = seed

        self.num_examples = num_examples
        self.noise_scale = noise_scale  # the parameter for the covariance matrix (variance)

        # coefficient of noise, lower means less effect of noise
        # only affects non root nodes
        self.noise_strength = noise_strength
        self.rng = np.random.default_rng(seed=seed)

        # Dealing with HTE inducing variables
        if num_hte_parents < 0:
            num_hte_parents = 0
        self.num_hte_parents = num_hte_parents
        self.num_used_hte_variables = 0
        self.hte_from_mediators = hte_from_mediators

        self.noise = None
        self.coeffs = None

        self.contains_heterogeneity = False

        self.time_debug = time_debug

        self.simple_nonlinear = simple_nonlinear

        self.normalize_coeffs = normalize_coeffs

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        random.seed(self.seed)

    def generate_data(self):
        self.set_seed(self.seed)

        # update fields based on graph (if graph has changed)
        self.update_from_graph()

        # generate coefficients from adjacency
        self._generate_coeffs()

        # generate noise terms
        self._generate_noise()

        # generate data
        df = self._generate()

        return df

    def update_from_graph(self):
        self.num_variables = self.graph_generator.num_variables
        self.graph = self.graph_generator.graph
        self.relabels = self.graph_generator.relabels
        self.relabels_reverse = self.graph_generator.relabels_reverse
        self.t_position = self.graph_generator.t_position
        self.y_position = self.graph_generator.y_position
        self.paths_from_t_to_y = self.graph_generator.paths_from_t_to_y
        # 2021-04-27 - removed all simple paths
        self.desc_t = self.graph_generator.desc_t
        self.anc_y = self.graph_generator.anc_y

    def _generate_coeffs(self):
        zero_one_adjacency = self.graph_generator.adjacency
        self.coeffs = self.graph_generator.adjacency.copy()
        for i in range(self.graph_generator.num_variables):
            for j in range(i + 1, self.graph_generator.num_variables):
                if np.isclose(zero_one_adjacency[i, j], 1):
                    # ----------------------------------------------------------------
                    # 2021-01-06 - messing with coefficient generation (to prevent 0)
                    # ----------------------------------------------------------------
                    self.coeffs[i, j] = self.rng.uniform(0.25, 1) * self.rng.choice([-1, 1], p=[0.5, 0.5])
                    # self.coeffs[i, j] = self.rng.uniform(-1, 1)

    def _generate_noise(self):
        start = time()

        # ----------------------------------------------------------------
        # 2021-01-06 - messing with noise generation
        # ----------------------------------------------------------------

        # cov = np.ones((self.num_variables, self.num_variables))
        # for i in range(self.num_variables):
        #     for j in range(self.num_variables):
        #         if i != j:
        #             cov[i, j] = self.noise_scale
        #
        # non_zero_sum_col_adjacency = self.coeffs.sum(axis=0) != 0
        # self.noise = self.rng.multivariate_normal(np.zeros(self.num_variables), cov=cov, size=self.num_examples)
        # self.noise[:, non_zero_sum_col_adjacency] = np.sqrt(self.noise_strength) * self.noise[:,
        #                                                                                 non_zero_sum_col_adjacency]

        self.noise = self.rng.normal(loc=0, scale=self.noise_scale, size=(self.num_examples, self.num_variables))
        non_zero_sum_col_adjacency = self.coeffs.sum(axis=0) != 0
        # self.noise = self.noise * self.noise_strength
        self.noise[:, non_zero_sum_col_adjacency] = self.noise_strength * self.noise[:, non_zero_sum_col_adjacency]

        end = time()
        if self.time_debug:
            print(f"Generate noise time: {end - start}")

    def _generate_data(self, counterfactual=False,
                       test_flag=False):

        resample = self.normalize_coeffs

        self.set_seed(self.seed)  # to guarantee cf? - seems like this works
        t_probs = self.rng.uniform(0, 1, size=self.num_examples)

        data = np.zeros((self.num_examples, self.num_variables))
        for i, node_id in enumerate(self.relabels):
            node = self.relabels[node_id]
            parents = list(self.graph.predecessors(node))
            data_col = self.noise[:, i].copy()  # copy so i don't accidentally change noise

            for parent in parents:
                parent_id = self.relabels_reverse[parent]
                data_col = data_col + self.coeffs[parent_id, i] * data[:, parent_id]

            # resample coeffs to get bounded values
            if resample:
                data_std = np.std(data_col)
                data_col = self.noise[:, i].copy()
                for parent in parents:
                    parent_id = self.relabels_reverse[parent]
                    data_col = data_col + self.coeffs[parent_id, i] / data_std * data[:, parent_id]

            if node == "T":  # nonlinear transform to get a binary treatment
                data_col = sigmoid(data_col)
                data_col = (data_col >= t_probs).astype(int)
                if counterfactual:
                    data_col = np.abs(data_col - 1)
            elif node == "Y":
                data_col += self._hte_induction(data, data_col, parents)

            # 2021-04-23 - Added hte from mediator parents
            # elif check_mediator(node, self.paths_from_t_to_y) and self.hte_from_mediators:
            elif not test_flag and self._check_mediator(node) and self.hte_from_mediators:
                data_col += self._hte_induction(data, data_col, parents)

            if self.simple_nonlinear and node != "T":
                data[:, i] = np.tanh(data_col)
            else:
                data[:, i] = data_col

        return data

    def _check_mediator(self, p):
        return p in self.desc_t and p in self.anc_y

    def _hte_induction(self, data, data_col, parents):
        # make these class variables so that I can use them again for CF?
        # might be unnecessary since I set the seed now in _generate_linear
        # the only thing this would prevent is using the same graph with different heterogeneity
        # (which I dont think matters?)

        # get all non_treatment parents of T
        desc_t = nx.algorithms.dag.descendants(self.graph, "T")
        non_treatment_parents = [p for p in parents if
                                 p != "T" and
                                 not self._check_mediator(p) and
                                 p not in desc_t]
        mediating_parents = [p for p in parents if self._check_mediator(p)]

        # ----------------------------------------------------------------
        # randomly get heterogeneity inducing parents
        # 2021-03-19 - For now, set only 1 HTE inducing parents
        # 2021-04-23 - Added selection of number using parameters
        # ----------------------------------------------------------------
        # num_hte_parents = min(self.num_hte_parents, self.rng.integers(len(non_treatment_parents) + 1))
        # num_hte_parents = 1
        num_hte_parents = self.num_hte_parents

        which_parents = self.rng.permutation(len(non_treatment_parents))[:num_hte_parents]
        # Sample coefficients for interaction term
        extra_coeffs = np.zeros((len(non_treatment_parents), len(mediating_parents)))
        extra_coeffs[which_parents] = self.rng.uniform(
            -1, 1, size=(min(self.num_hte_parents, len(which_parents)), len(mediating_parents))
        )
        extra_coeffs = extra_coeffs[which_parents]
        hte_parents = [non_treatment_parents[p] for p in which_parents]
        # ["X0"]
        if len(hte_parents) > 0:
            self.contains_heterogeneity = True
        # Add interaction terms to "data_col"
        # ----------------------------------------------------------------
        # 2021-07-16 I think I messed up hte induction for mediators (used to be self.t_position)
        # ----------------------------------------------------------------
        interaction = np.zeros(data.shape[0])
        for j, parent in enumerate(hte_parents):
            parent_id = self.relabels_reverse[parent]
            # interaction = data[:, self.t_position] * coeff * data[:, parent_id]  # could do (t - 0.5) also
            for i, med_par in enumerate(mediating_parents):
                med_parent_id = self.relabels_reverse[med_par]
                interaction += interaction + data[:, med_parent_id] * extra_coeffs[j, i] * data[:, parent_id]
        interaction_std = np.std(interaction)
        interaction = np.zeros(data.shape[0])
        for j, parent in enumerate(hte_parents):
            parent_id = self.relabels_reverse[parent]
            # interaction = data[:, self.t_position] * coeff * data[:, parent_id]  # could do (t - 0.5) also
            for i, med_par in enumerate(mediating_parents):
                med_parent_id = self.relabels_reverse[med_par]
                interaction += interaction + data[:, med_parent_id] * extra_coeffs[j, i] / interaction_std * data[:,
                                                                                                             parent_id]
        data_col = data_col + interaction

        return data_col

    def _generate(self):
        start = time()

        # TODO: select data generator - for now its linear
        data = self._generate_data()
        cf_data = self._generate_data(counterfactual=True)

        x_cols = [i for i in range(self.num_variables) if i != self.t_position and i != self.y_position]
        x = data[:, x_cols]
        y = data[:, self.y_position]
        t = data[:, self.t_position]

        treated = t == 1
        control = ~treated
        effect = np.zeros(self.num_examples)
        effect[treated] = data[treated, self.y_position] - cf_data[treated, self.y_position]
        effect[control] = cf_data[control, self.y_position] - data[control, self.y_position]

        dat_dict = dict()
        x_counter = 0
        for j in range(data.shape[1]):
            if j == self.t_position:
                dat_dict["t"] = data[:, j]
            elif j == self.y_position:
                dat_dict["y"] = data[:, j]
            else:
                dat_dict[f"x{x_counter}"] = data[:, j]
                x_counter += 1
        # for j in range(x.shape[1]):
        #     dat_dict[f"x{j}"] = x[:, j]
        # dat_dict["t"] = t
        # dat_dict["y"] = y
        dat_dict["cf_y"] = cf_data[:, self.y_position]
        dat_dict["effect"] = effect
        df = pd.DataFrame(dat_dict)

        end = time()
        if self.time_debug:
            print(f"Generate data time: {end - start}")

        return df


class FullGeneration:
    def __init__(self,
                 num_examples=1000,
                 num_variables=3,
                 edge_prob=0.1,
                 noise_scale=0.1,
                 noise_strength=0.1,
                 mediator=True,
                 mediator_chain_length=0,
                 num_hte_parents=1,
                 hte_from_mediators=False,
                 confounder=True,
                 seed=724,
                 time_debug=False,
                 simple_nonlinear=False,
                 normalize_coeffs=True):
        self.num_examples = num_examples
        self.num_variables = num_variables
        self.edge_prob = edge_prob
        self.noise_scale = noise_scale
        self.noise_strength = noise_strength
        self.seed = seed
        self.time_debug = time_debug

        self.mediator = mediator
        self.mediator_chain_length = mediator_chain_length

        self.confounder = confounder

        self.num_hte_parents = num_hte_parents
        self.hte_from_mediators = hte_from_mediators

        self.simple_nonlinear = simple_nonlinear

        self.normalize_coeffs = normalize_coeffs

        self.graph_generator = GraphGeneration(num_variables=self.num_variables, edge_prob=self.edge_prob,
                                               mediator=self.mediator, mediator_chain_length=self.mediator_chain_length,
                                               seed=self.seed, time_debug=self.time_debug, confounder=confounder)
        self.data_generator = DataGeneration(self.graph_generator, num_examples=self.num_examples,
                                             noise_scale=self.noise_scale,
                                             noise_strength=self.noise_strength,
                                             hte_from_mediators=self.hte_from_mediators,
                                             num_hte_parents=self.num_hte_parents,
                                             seed=self.seed,
                                             time_debug=self.time_debug, simple_nonlinear=self.simple_nonlinear,
                                             normalize_coeffs=self.normalize_coeffs)

        self.contains_confounding = False
        self.contains_mediation = False
        self.contains_heterogeneity = False

        self.discard = False

    def set_seed(self, seed):
        random.seed(self.seed)
        self.seed = seed

    def generate_data(self):
        self.graph_generator.generate_graph()

        self.data_generator.discard = self.graph_generator.discard

        self.mediator = self.graph_generator.mediator

        df = self.data_generator.generate_data()

        self.discard = self.data_generator.discard

        self.contains_confounding = self.graph_generator.contains_confounding
        self.contains_mediation = self.graph_generator.contains_mediation
        self.contains_heterogeneity = self.data_generator.contains_heterogeneity

        return df

    def get_params(self):
        return {
            "contains_confounding": self.contains_confounding,
            "contains_mediation": self.contains_mediation,
            "contains_heterogeneity": self.contains_heterogeneity,
            "num_examples": self.num_examples,
            "num_variables": self.num_variables,
            "edge_prob": self.edge_prob,
            "noise_scale": self.noise_scale,
            "noise_strength": self.noise_strength,
            "seed": self.seed,
            "mediator": self.mediator,
            "mediator_chain_length": self.mediator_chain_length,
            "num_hte_parents": self.num_hte_parents,
            "hte_from_mediators": self.hte_from_mediators,
        }
