import os
import random
from collections.abc import Sequence
from itertools import zip_longest

import numpy as np
import pandas as pd
from IPython.core.magic import Magics, line_magic, magics_class
from IPython.display import HTML, Markdown, display

from .constants import LATEX_MACROS

__all__ = ["set_random_seed", "TflWorkshopMagic", "display_dataframes_side_by_side"]


def set_random_seed(seed: int = 16) -> None:
    """Sets random seed (as name suggests...)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import pytorch_lightning as pl

        pl.seed_everything(seed)
    except ImportError:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass

    os.environ["PYTHONHASHSEED"] = str(seed)


@magics_class
class TflWorkshopMagic(Magics):
    def __init__(self, shell):
        super().__init__(shell)

    @line_magic
    def set_random_seed(self, seed: str):
        seed = int(seed) if seed else 16
        set_random_seed(seed)

    @line_magic
    def load_latex_macros(self, line):
        display(Markdown(LATEX_MACROS))

    @line_magic
    def view_hint(self, path: os.PathLike):
        display(Markdown(path))

    @line_magic
    def presentation_style(self, style_file: str):
        """Apply the styles to the notebook (outside presentation mode).
        **NOTE**: Has to be the last command in a cell.

        :param style_file: Relative path to the CSS file containing
            the style that will be applied to the notebook cells.
            Defaults to `rise.css`
        """
        # NOTE: unfortunately, default values kwargs are not possible here
        # because ipython actively sends an empty string as the value of the argument if nothing is passed...
        if not style_file:
            style_file = "rise.css"
        with open(style_file) as f:
            styles = f.read()
        return HTML(f"<style>{styles}</style>")


def display_dataframes_side_by_side(
    dataframes: Sequence[pd.DataFrame],
    captions: Sequence = (),
):
    """Display pandas dataframes side by side in a jupyter notebook.

    Inspired by: https://stackoverflow.com/a/64323280
    """
    if len(captions) > len(dataframes):
        raise ValueError(
            f"There are more captions than dataframes. "
            f"Got {len(captions)} captions and {len(dataframes)} dataframes.",
        )

    # NOTE: we previously had a widgets based solution, but it messes something
    # up in the notebooks state in a very evil and subtle way.
    output = ""
    for caption, df in zip_longest(captions, dataframes):
        output += (
            df.style.set_table_attributes("style='display:inline'")
            .set_caption(caption or "")
            .to_html()
        )
        output += "\xa0\xa0\xa0"

    display(HTML(output))
