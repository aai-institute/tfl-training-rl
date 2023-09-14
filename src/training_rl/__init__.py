__version__ = "1.0.0"
from .nb_utils import TflWorkshopMagic


def load_ipython_extension(ipython):
    """Defining this function here allows to load the TflWorkshopMagic extension
    by using %load_ext training_rl
    """
    ipython.register_magics(TflWorkshopMagic)
