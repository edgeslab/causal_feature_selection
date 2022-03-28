from pyCausalFS.CBD.MBs.HITON.HITON_PC import HITON_PC
from pyCausalFS.CBD.MBs.pc_simple import pc_simple
from pyCausalFS.CBD.MBs.PCMB.getPC import getPC
from pyCausalFS.CBD.MBs.semi_HITON.semi_HITON_PC import semi_HITON_PC
from pyCausalFS.LSL.MBs.CMB.CMB import CMB
from pyCausalFS.LSL.MBs.PCDbyPCD import PCDbyPCD
from pyCausalFS.LSL.MBs.MBbyMB import MBbyMB
import numpy as np
from time import time
from collections import defaultdict

orients_edges_bool = defaultdict(lambda: False,
                                 {
                                     CMB: True,
                                     PCDbyPCD: True,
                                     MBbyMB: True,
                                 })


class PCSelect:

    def __init__(self, pc_alg, timer=False):

        self.timer = timer

        self.pc_alg_lookup = {
            "pc_simple": pc_simple,
            "get_pc": getPC,
            "getPC": getPC,
            "hiton_pc": HITON_PC,
            "hitonpc": HITON_PC,
            "semi_hiton_pc": semi_HITON_PC,
            "semi hiton pc": semi_HITON_PC,
            "cmb": CMB,
            "pcdbypcd": PCDbyPCD,
            "mbbymb": MBbyMB,
        }

        if isinstance(pc_alg, str):
            if pc_alg.lower() in self.pc_alg_lookup:
                self.pc_alg = self.pc_alg_lookup[pc_alg.lower()]
            else:
                pc_alg = "pc_simple"
                self.pc_alg = self.pc_alg_lookup[pc_alg.lower()]
        else:
            self.pc_alg = pc_alg

        self.t_col = "t"
        self.y_col = "y"

        self.orients_edges = orients_edges_bool[self.pc_alg]

    def output_pc(self, input_data, alpha=0.05, binary_data=False):
        start = time()

        data = input_data[[i for i in input_data.columns if i != "effect"]]

        y_col = np.where(data.columns == self.y_col)[0][0]
        t_col = np.where(data.columns == self.t_col)[0][0]

        # print(data.shape, y_col, binary_data)
        y_out = self.pc_alg(data, y_col, alpha, binary_data)
        if self.timer:
            end = time()
            print(f"Elapsed time finding pc(Y): {end - start:0.3f}")
            start = time()

        t_out = self.pc_alg(data, t_col, alpha, binary_data)
        if self.timer:
            end = time()
            print(f"Elapsed time finding pc(T): {end - start:0.3f}")

        # if using CMB for example, it produces parents and children
        if self.orients_edges:
            y_parents, y_children, _, y_undirected, _ = y_out
            t_parents, t_children, _, t_undirected, _ = t_out

            y_parents_cols = list(data.columns[y_parents])
            y_children_cols = list(data.columns[y_children])
            y_undirected_cols = list(data.columns[y_undirected])
            t_parents_cols = list(data.columns[t_parents])
            t_children_cols = list(data.columns[t_children])
            t_undirected_cols = list(data.columns[t_undirected])

            return y_parents_cols, y_children_cols, y_undirected_cols, t_parents_cols, t_children_cols, t_undirected_cols
        else:
            y_pc = y_out[0]
            t_pc = t_out[0]

            y_pc_cols = data.columns[y_pc]
            t_pc_cols = data.columns[t_pc]

            return list(y_pc_cols), list(t_pc_cols)

    def output_input_pc(self, input_data, input_var, alpha=0.05, binary_data=False, known_parents=None):
        start = time()

        data = input_data[[i for i in input_data.columns if "effect" not in i]]

        input_col = np.where(data.columns == input_var)[0][0]
        input_var_out = self.pc_alg(data, input_col, alpha, binary_data)
        if self.timer:
            end = time()
            print(f"Elapsed time finding pc({input_var}): {end - start:0.3f}")
            start = time()

        # if using CMB for example, it produces parents and children
        if self.orients_edges:
            input_parents, input_children, _, input_undirected, _ = input_var_out

            input_parents_cols = list(data.columns[input_parents])
            input_children_cols = list(data.columns[input_children])
            input_undirected_cols = list(data.columns[input_undirected])
            if known_parents is not None:
                input_parents_cols = input_parents_cols + known_parents
            return input_parents_cols, input_children_cols, input_undirected_cols
        else:
            input_pc = input_var_out[0]
            input_pc_cols = data.columns[input_pc]

            if known_parents is not None:
                return list(input_pc_cols) + known_parents
            return list(input_pc_cols)
