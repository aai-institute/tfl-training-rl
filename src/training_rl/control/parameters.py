from dataclasses import dataclass

__all__ = ["MassSpringDamperParameters", "InvertedPendulumParameters"]


@dataclass
class MassSpringDamperParameters:
    """Attributes
    m: Mass of the object.
    c: Damping coefficient.
    k: Spring coefficient.
    """

    m: float = 0.51
    c: float = 1.52
    k: float = 50.15


@dataclass
class InvertedPendulumParameters:
    """Attributes
    m: Mass of the pendulum.
    M: Mass of the cart.
    l: Length of the pendulum.
    g: Gravitational constant.

    """

    m: float = 2
    M: float = 10
    l: float = 0.3
    g: float = 9.81
