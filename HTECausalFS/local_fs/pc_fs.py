import itertools
import numpy as np
from HTECausalFS.local_fs._edge_orientation import EdgeOrientation
from HTECausalFS.local_fs._pc import *
from fcit import fcit  # https://github.com/kjchalup/fcit -> pip install fcit
from time import time
from sklearn.linear_model import LogisticRegression


class PCFeatureSelect:

    def __init__(self, pc_alg="pc_simple", edge_alg="reci", adjust_method="parents", use_propensity=False,
                 propensity_model=LogisticRegression, propensity_params=None, alpha=0.05, binary_data=False,
                 check_colliders=False, timer=False):

        self.use_propensity = use_propensity
        if propensity_params is not None:
            self.propensity_model = propensity_model(**propensity_params)
        else:
            self.propensity_model = LogisticRegression()

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

        if adjust_method == "parents":
            self.adjust_method = self.adjustment_set_parents
        elif adjust_method == "set":
            self.adjust_method = self.adjustment_set_diff
        elif adjust_method == "independence" or adjust_method == "ind":
            self.adjust_method = self.adjustment_set_selection
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

    # ----------------------------------------------------------------
    # TODO: combine set_diff and set_independence...
    # if T dep Y given Z then return Z automatically
    # ----------------------------------------------------------------
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
        pass

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
