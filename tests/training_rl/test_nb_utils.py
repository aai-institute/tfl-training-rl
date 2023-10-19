import numpy as np

from training_rl.nb_utils import set_random_seed


# the suggested naming convention for unit tests is test_method_name_testDescriptionInCamelCase
# this leads to a nicely readable output of pytest
def test_set_random_seed():
    set_random_seed(16)
    assert np.random.get_state()[1][0] == 16
