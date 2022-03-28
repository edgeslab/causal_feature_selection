import itertools
import numpy as np
from HTECausalFS.local_fs._edge_orientation import EdgeOrientation
from HTECausalFS.local_fs._pc import *
from fcit import fcit  # https://github.com/kjchalup/fcit -> pip install fcit
from time import time
from sklearn.linear_model import LogisticRegression
import networkx as nx
from collections import defaultdict


class PCFeatureSelect:

    def __init__(self, pc_alg="pc_simple", edge_alg="reci", adjust_method="independence", use_propensity=False,
                 propensity_model=LogisticRegression, propensity_params=None, alpha=0.05, binary_data=False,
                 check_colliders=False, timer=False):

        self.use_propensity = use_propensity
        if propensity_params is not None:
            self.propensity_model = propensity_model(**propensity_params)
        else:
            self.propensity_model = propensity_model()

        self.timer = timer

        self.pc_alg = pc_alg
        self.edge_alg = edge_alg

        self.pc = PCSelect(pc_alg, timer=self.timer)
        self.eo = EdgeOrientation(edge_alg, timer=self.timer)

        self.data = None
        self.alpha = alpha
        self.binary_data = binary_data
        self.check_colliders = check_colliders

        self.t_binary = True

        self.y_col = "y"
        self.t_col = "t"

        self.adjustment_set = None

        self.adjust_method_str = adjust_method
        if adjust_method == "parents":
            self.adjust_method = self.adjustment_set_parents
        elif adjust_method == "set":
            self.adjust_method = self.adjustment_set_diff
        elif adjust_method == "independence" or adjust_method == "ind":
            self.adjust_method = self.adjustment_set_selection
        elif adjust_method == "o-set" or adjust_method == "o_set" or adjust_method == "o set":
            self.adjust_method = self.o_set_approximation
        elif adjust_method == "nonforbidden":
            self.adjust_method = self.o_set_approximation
        elif adjust_method == "independence_old" or adjust_method == "ind_old":
            self.adjust_method = self.adjustment_set_independence
        else:
            self.adjust_method = self.adjustment_set_parents

    def adjustment_set_parents(self, y_parents, y_children, t_parents, t_children):
        if len(t_parents) < 1:
            return t_children
        return t_parents

    def adjustment_set_diff(self, y_parents, y_children, t_parents, t_children):
        # if there is more than 1 mediator then this doesn't work!
        adjustment_cols = list(set(t_parents).union(set(y_parents).difference(set(t_children))))

        return adjustment_cols

    def adjustment_set_independence(self, y_parents, y_children, t_parents, t_children):

        Z = [i for i in t_parents]
        M = []
        y_data = self.data[[self.y_col]].values
        t_data = self.data[[self.t_col]].values
        for child in t_children:
            if child in y_children:
                continue
            child_data = self.data[[child]].values
            Z_data = self.data[Z].values
            comb_data = np.hstack((Z_data, child_data))
            pval = fcit.test(t_data, y_data, comb_data)
            if pval <= self.alpha:
                if child not in M:
                    M.append(child)

        for parent in y_parents:
            if parent in t_parents or parent in M:
                continue
            child_data = self.data[[parent]].values
            Z_data = self.data[Z].values
            comb_data = np.hstack((Z_data, child_data))
            pval = fcit.test(t_data, y_data, comb_data)
            if pval <= self.alpha:
                if parent not in M:
                    M.append(parent)
            else:
                if parent not in Z:
                    Z.append(parent)

        # print(Z)
        return Z

    def o_set_approximation(self, y_parents, y_children, t_parents, t_children):
        # initialize data col number map
        # data_col_map = {col: i for i, col in enumerate(self.data.columns)}
        all_known_parents = defaultdict(lambda: None, {self.t_col: t_parents, self.y_col: y_parents})

        dn = set(y_children)
        Z = set(t_children)
        V = {self.t_col}.union(t_children)

        E = set([(self.t_col, j) for j in Z])
        E = E.union([(j, self.t_col) for j in t_parents])
        E = E.union([(j, self.y_col) for j in y_parents])
        E = E.union([(self.y_col, j) for j in y_children])
        edge_count = {e: 1 for e in E}
        while len(Z) > 0:
            z = Z.pop()
            # find pa(z) and ch(z)
            pa_z, ch_z = self.get_input_parent_children(self.data, z, known_parents=all_known_parents[z])

            dn = dn.union([z])
            Z = Z.union(ch_z).difference(dn)
            V = V.union(Z)
            #     E = E.union(set([(z, z_c) for z_c in ch_z]))
            for z_c in ch_z:
                e = (z, z_c)
                #         E.add(e)
                edge_count.setdefault(e, 0)
                edge_count[e] += 1
            for z_p in pa_z:
                e = (z_p, z)
                #         E. add(e)
                edge_count.setdefault(e, 0)
                edge_count[e] += 1

        remove_edges = []
        for e in edge_count:
            r_e = (e[1], e[0])
            if r_e in edge_count:
                # print(e, r_e)
                if edge_count[r_e] > edge_count[e]:
                    # edge_count.popitem(e)
                    # del edge_count[e]
                    remove_edges.append(e)
                else:
                    # edge_count.popitem(r_e)
                    # del edge_count[e]
                    remove_edges.append(r_e)
        edge_count = {e: val for e, val in zip(edge_count.keys(), edge_count.values()) if
                      edge_count not in remove_edges}
        G = nx.DiGraph()
        G.add_edges_from(edge_count.keys())
        if self.t_col not in G.nodes():
            G.add_node(self.t_col)
        if self.y_col not in G.nodes():
            G.add_node(self.y_col)

        if self.adjust_method_str == "nonforbidden":
            return self.non_forbidden(G)
        return self.optimal_adjust(G)

    def adjustment_set_selection(self, y_parents, y_children, t_parents, t_children):
        def med_dfs(col, M, paM, known_parents=None, known_children=None):
            if col in y_parents:
                M.add(M)

            if known_parents is not None and known_children is not None:
                parents = known_parents
                children = known_children
            else:
                parents, children = self.get_input_parent_children(self.data, col, known_parents=known_parents)

            paM = paM.union(parents)
            for ch in children:
                if ch == "y":
                    continue
                if ch not in t_parents:
                    M.add(ch)
                    if ch not in y_parents:
                        # recursive call
                        med_dfs(ch, M, paM, known_parents=parents)

        # initialize Z
        Z = set(y_parents).union(t_parents)
        M = set()
        paM = set()

        med_dfs(self.t_col, M, paM, known_parents=y_parents, known_children=t_children)

        Z = Z.union(paM).difference(M)

        return list(Z)

    def get_adjustment_set(self, data, y_col="y", t_col="t"):

        self.y_col = y_col
        self.t_col = t_col
        self.pc.y_col = y_col
        self.pc.t_col = t_col
        self.eo.y_col = y_col
        self.eo.t_col = t_col

        y_parents, y_children, t_parents, t_children = self.get_p_and_c(data)

        adjustment_set = self.adjust_method(y_parents, y_children, t_parents, t_children)

        final_adjustment_cols = [i for i in adjustment_set if i != self.y_col and i != self.t_col]

        return final_adjustment_cols

    def get_p_and_c(self, data):
        # check if T is binary
        if np.unique(data[self.t_col]).shape[0] == 2:
            self.t_binary = True
        self.data = data.copy()

        if self.use_propensity:
            all_cols = [i for i in data.columns if i != self.t_col and i != self.y_col and i != "effect"]
            x = data[all_cols].values
            t = data[self.t_col].values

            self.propensity_model.fit(x, t)
            t_prop = self.propensity_model.predict(x)
            self.data[self.t_col] = t_prop

        return self._get_p_and_c()

    def _get_p_and_c(self):

        # if using CMB, only orient edges for undirected edges
        if self.pc.orients_edges:
            output_package = self.pc.output_pc(self.data, self.alpha, self.binary_data)
            y_parents, y_children, y_undirected_cols = output_package[0], output_package[1], output_package[2]
            t_parents, t_children, t_undirected_cols = output_package[3], output_package[4], output_package[5]

            if len(y_undirected_cols) > 0 or len(t_undirected_cols) > 0:
                y_pc = y_parents + y_children + y_undirected_cols
                t_pc = t_parents + t_children + y_undirected_cols
                y_parents, y_children, t_parents, t_children = self.eo.orient_edges(self.data, y_pc, t_pc,
                                                                                    known_parents_y=y_parents,
                                                                                    known_parents_t=t_parents,
                                                                                    known_children_y=y_children,
                                                                                    known_children_t=t_children)
        else:
            y_pc, t_pc = self.pc.output_pc(self.data, self.alpha, self.binary_data)
            known_parents_y, known_parents_t = None, None
            known_children_y, known_children_t = None, None
            if self.check_colliders:
                known_parents_y, known_parents_t = self._independence_parents(y_pc, t_pc)
                if len(known_parents_y) > 0:
                    known_children_y = [i for i in y_pc if i not in known_parents_y]
                if len(known_parents_t) > 0:
                    known_children_t = [i for i in t_pc if i not in known_parents_t]
            y_parents, y_children, t_parents, t_children = self.eo.orient_edges(self.data, y_pc, t_pc,
                                                                                known_parents_y=known_parents_y,
                                                                                known_parents_t=known_parents_t,
                                                                                known_children_y=known_children_y,
                                                                                known_children_t=known_children_t)

        return y_parents, y_children, t_parents, t_children

    def get_input_parent_children(self, data, input_var, known_parents):

        if self.pc.orients_edges:
            parents, children, input_undirected = self.pc.output_input_pc(data, input_var, self.alpha,
                                                                          self.binary_data,
                                                                          known_parents=known_parents)
            if len(input_undirected) > 0:
                pc = parents + children + input_undirected
                parents, children = self.eo.orient_input_edge(data, pc, input_var,
                                                              known_parents=parents,
                                                              known_children=children)
        else:
            pc = self.pc.output_input_pc(data, input_var, self.alpha, self.binary_data, known_parents=known_parents)

            known_parent_colliders = []
            if self.check_colliders:
                known_parent_colliders = self._independence_parents_input(pc, input_var)
            known_parent_colliders = known_parent_colliders + known_parents
            known_children_colliders = None
            if len(known_parent_colliders) > 0:
                known_children_colliders = [i for i in pc if i not in pc]

            parents, children = self.eo.orient_input_edge(data, pc, input_var, known_parents=known_parent_colliders,
                                                          known_children=known_children_colliders)

        return parents, children

    def _independence_parents(self, y_pc, t_pc):
        start = time()

        data = self.data

        y_data = data[[self.y_col]].values
        t_data = data[[self.t_col]].values
        known_parents_y = []
        known_parents_t = []
        y_pc_combs = itertools.combinations(y_pc, 2)
        for comb in y_pc_combs:
            data1 = data[[comb[0]]].values
            data2 = data[[comb[1]]].values
            pval_u = fcit.test(data1, data2)
            pval_c = fcit.test(data1, data2, y_data)

            if pval_u <= self.alpha < pval_c:
                for c in comb:
                    if c not in known_parents_y:
                        known_parents_y.append(c)

        if self.timer:
            end = time()
            print(f"Elapsed time for independence of pc(Y) structure: {end - start:0.3f}")
            start = time()

        t_pc_combs = itertools.combinations(t_pc, 2)
        for comb in t_pc_combs:
            data1 = data[[comb[0]]].values
            data2 = data[[comb[1]]].values
            pval_u = fcit.test(data1, data2)
            pval_c = fcit.test(data1, data2, t_data)

            if pval_u <= self.alpha < pval_c:
                for c in comb:
                    if c not in known_parents_t:
                        known_parents_t.append(c)

        if self.timer:
            end = time()
            print(f"Elapsed time for independence of pc(T) structure: {end - start:0.3f}")

        return known_parents_y, known_parents_t

    def _independence_parents_input(self, pc, input_var):

        start = time()

        data = self.data

        y_data = data[[input_var]].values
        known_parents_y = []
        y_pc_combs = itertools.combinations(pc, 2)
        for comb in y_pc_combs:
            data1 = data[[comb[0]]].values
            data2 = data[[comb[1]]].values
            pval_u = fcit.test(data1, data2)
            pval_c = fcit.test(data1, data2, y_data)

            if pval_u <= self.alpha < pval_c:
                for c in comb:
                    if c not in known_parents_y:
                        known_parents_y.append(c)

        if self.timer:
            end = time()
            print(f"Elapsed time for independence of pc({input_var}) structure: {end - start:0.3f}")

        return known_parents_y

    def optimal_adjust(self, graph, treatment="t", outcome="y"):
        # all_paths_from_t_to_y = list(nx.all_simple_paths(graph, treatment, outcome))
        # posscn = set().union(*all_paths_from_t_to_y)
        # posscn.add(outcome)
        # posscn.remove(treatment)
        # possde = [nx.descendants(graph, node) for node in posscn]
        # possde = set().union(*possde)
        # forbidden_set = possde.union(treatment)

        all_paths_from_t_to_y = list(nx.all_simple_paths(graph, treatment, outcome))
        posscn = set().union(*all_paths_from_t_to_y)
        # posscn.add(outcome)
        # posscn.remove(treatment)
        possde = [nx.descendants(graph, node) for node in posscn]
        possde = set().union(*possde)
        forbidden_set = possde.union(treatment)

        pacn = [list(graph.predecessors(node)) for node in posscn]
        pacn = set().union(*pacn)

        adjustment_set = pacn.difference(forbidden_set)

        non_forbidden_set = set(list(graph.nodes)).difference(forbidden_set)
        self.adjustment_set = list(adjustment_set)

        return self.adjustment_set

    def non_forbidden(self, graph, treatment="t", outcome="y"):
        all_paths_from_t_to_y = list(nx.all_simple_paths(graph, treatment, outcome))
        forbidden_set = set().union(*all_paths_from_t_to_y)
        possde = [nx.descendants(graph, node) for node in forbidden_set]
        possde = set().union(*possde)
        forbidden_set = forbidden_set.union(possde)

        treatment_nodes = [treatment]
        outcome_nodes = [outcome]
        forbidden_set = forbidden_set.union(treatment_nodes).union(outcome_nodes)

        all_nodes = set(graph.nodes())

        adjustment_set = all_nodes.difference(forbidden_set)
        self.adjustment_set = list(adjustment_set)

        return self.adjustment_set
