import cdt
from cdt.causality.pairwise import IGCI, RECI, CDS
from cdt.causality.graph import PC
import networkx as nx
import numpy as np
from time import time


class EdgeOrientation:

    def __init__(self, edge_alg, timer=False):

        self.timer = timer

        self.non_binary_eo = ["cds", CDS, "pc", PC]

        self.edge_alg_lookup = {
            "igci": IGCI,
            "reci": RECI,
            "cds": CDS,
            "pc": PC
        }

        self.graph_based = ["pc", PC]

        self.t_binary = True

        if isinstance(edge_alg, str):
            if edge_alg.lower() in self.edge_alg_lookup:
                self.edge_alg = self.edge_alg_lookup[edge_alg.lower()]
                self.edge_alg_name = edge_alg.lower()
            else:
                edge_alg = "cds"
                self.edge_alg = self.edge_alg_lookup[edge_alg.lower()]
                self.edge_alg_name = edge_alg.lower()
        else:
            self.edge_alg = edge_alg
            self.edge_alg_name = edge_alg

        self.can_binary = False
        if self.edge_alg in self.non_binary_eo:
            self.can_binary = True

    def orient_edges(self, data, y_pc, t_pc, known_parents_y=None, known_parents_t=None, known_children_y=None,
                     known_children_t=None):
        start = time()

        obj = self.edge_alg()

        # ----------------------------------------------------------------
        # Y PC
        # ----------------------------------------------------------------
        y_dat = data[["y"] + y_pc]
        y_graph = nx.Graph()
        for col in y_pc:
            if known_parents_y is not None and col in known_parents_y:
                continue
            if known_children_y is not None and col in known_children_y:
                continue
            y_graph.add_edge("y", col)

        if self.edge_alg_name in self.graph_based:
            y_output = obj.predict(y_dat, nx.Graph(y_graph))
        else:
            y_output = obj.orient_graph(y_dat, nx.Graph(y_graph))

        y_parents = []
        for parent in y_output.predecessors("y"):
            y_parents.append(parent)
        y_children = []
        for child in y_output.successors("y"):
            y_children.append(child)

        if known_parents_y is not None:
            y_parents = y_parents + known_parents_y
        if known_children_t is not None:
            y_children = y_children + known_children_t

        if self.timer:
            end = time()
            print(f"Elapsed time for orienting Y structure: {end - start:0.3f}")
            start = time()

        # ----------------------------------------------------------------
        # T PC
        # ----------------------------------------------------------------

        # check if T is binary
        if np.unique(data["t"]).shape[0] == 2:
            self.t_binary = True

        # if T is binary and we cannot use edge orientation on binary data, then skip
        if self.t_binary and not self.can_binary:
            if known_parents_t is not None:
                t_parents = known_parents_t
            else:
                t_parents = t_pc
            if known_children_t is not None:
                t_children = known_children_t
            else:
                t_children = list(set(t_pc).difference(t_parents))
            return y_parents, y_children, t_parents, t_children

        t_dat = data[["t"] + t_pc]
        t_graph = nx.Graph()
        for col in t_pc:
            if known_parents_t is not None and col in known_parents_t:
                continue
            if known_children_t is not None and col in known_children_t:
                continue
            t_graph.add_edge("t", col)

        if self.edge_alg_name in self.graph_based:
            t_output = obj.predict(t_dat, nx.Graph(t_graph))
            # print(t_output.edges())
        else:
            t_output = obj.orient_graph(t_dat, nx.Graph(t_graph))

        t_parents = []
        for parent in t_output.predecessors("t"):
            t_parents.append(parent)
        t_children = []
        for child in t_output.successors("t"):
            t_children.append(child)

        if known_parents_t is not None:
            t_parents = t_parents + known_parents_t
        if known_children_t is not None:
            t_children = t_children + known_children_t

        if self.timer:
            end = time()
            print(f"Elapsed time for orienting T structure: {end - start:0.3f}")
            start = time()

        return y_parents, y_children, t_parents, t_children

    def orient_input_edge(self, data, pc, input_var, known_parents=None, known_children=None):
        start = time()

        obj = self.edge_alg()

        y_dat = data[[input_var] + pc]
        y_graph = nx.Graph()
        for col in pc:
            if known_parents is not None and col in known_parents:
                continue
            if known_children is not None and col in known_children:
                continue
            y_graph.add_edge(input_var, col)

        y_output = obj.orient_graph(y_dat, nx.Graph(y_graph))

        parents = []
        for parent in y_output.predecessors(input_var):
            parents.append(parent)
        children = []
        for child in y_output.successors(input_var):
            children.append(child)

        if known_parents is not None:
            parents = parents + known_parents
        if known_children is not None:
            children = children + known_children

        if self.timer:
            end = time()
            print(f"Elapsed time for orienting {input_var} structure: {end - start:0.3f}")

        return parents, children
