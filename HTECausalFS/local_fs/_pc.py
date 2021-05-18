from CBD.MBs.HITON.HITON_PC import HITON_PC
from CBD.MBs import pc_simple
from CBD.MBs.PCMB import getPC
from CBD.MBs.semi_HITON.semi_HITON_PC import semi_HITON_PC
import numpy as np
from time import time


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
        }

        if isinstance(pc_alg, str):
            if pc_alg.lower() in self.pc_alg_lookup:
                self.pc_alg = self.pc_alg_lookup[pc_alg.lower()]
            else:
                pc_alg = "pc_simple"
                self.pc_alg = self.pc_alg_lookup[pc_alg.lower()]
        else:
            self.pc_alg = pc_alg

    def output_pc(self, input_data, alpha=0.05, binary_data=False):
        start = time()

        data = input_data[[i for i in input_data.columns if i != "effect"]]

        y_col = np.where(data.columns == "y")[0][0]
        t_col = np.where(data.columns == "t")[0][0]
        # y_col = data.columns[np.where(data.columns == "y")[0][0]]
        # t_col = data.columns[np.where(data.columns == "t")[0][0]]

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

        input_pc = input_var_out[0]
        input_pc_cols = data.columns[input_pc]

        if known_parents is not None:
            return list(input_pc_cols) + known_parents
        return list(input_pc_cols)
