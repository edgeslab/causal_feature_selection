import numpy as np

from HTECausalFS.data_gen.long_chain import continuous_effect
from HTECausalFS.local_fs.pc_fs import PCFeatureSelect

data = continuous_effect(num_examples=5000, num_x=2, variance=1.0, seed=0)

pcfs = PCFeatureSelect(check_colliders=False, timer=True, pc_alg="pc_simple", edge_alg="cds")

y_parents = ["x0", "x1", "f0", "f1"]
y_children = ["g0", "g1"]
t_parents = ["a0", "a1", "x0", "x1"]
t_children = ["b0", "b1", "d0", "d1"]

pcfs.get_input_parent_children(data, "b0", known_parents=["t"])
