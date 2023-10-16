from IPython.display import Markdown, display
from numpy.typing import NDArray

__all__ = ["display_array"]


def display_array(name: str, array: NDArray) -> None:
    """Displays numpy arrays as latex bmatrix."""
    matrix = ""
    for row in array:
        try:
            for number in row:
                matrix += f"{number} &"
        except TypeError:
            matrix += f"{row} &"
        matrix = matrix[:-1] + r"\\"
    display(Markdown(rf"$${name} = \begin{{bmatrix}}{matrix}\end{{bmatrix}}$$"))
