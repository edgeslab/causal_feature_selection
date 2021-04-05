import numpy as np

from HTECausalFS.data_gen.long_chain import continuous_effect
from HTECausalFS.local_fs.pc_fs import PCFeatureSelect

data = continuous_effect(num_examples=5000, num_x=10, variance=1.0, seed=0)

pcfs = PCFeatureSelect(timer=True)

package = pcfs.get_p_and_c(data)
