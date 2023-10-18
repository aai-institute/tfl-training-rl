from dataclasses import dataclass

__all__ = ["MassSpringDamperParameters", "InvertedPendulumParameters"]


@dataclass
class MassSpringDamperParameters:
    """Attributes
    m: Mass of the object.
    c: Damping coefficient.
    k: Spring coefficient.
    gamma: Motor gear factor.
    """

    m: float = 0.51
    c: float = 1.52
    k: float = 50.15
    gamma: float = 2.0


@dataclass
class InvertedPendulumParameters:
    """Attributes
    m: Mass of the pendulum.
    M: Mass of the cart.
    l: Length of the pendulum.
    g: Gravitational constant.
    mu_c: Cart friction constant.
    mu_p: Pendulum friction constant.
    gamma: Motor gear factor.
    """

    m: float = 10
    M: float = 10
    l: float = 0.3
    g: float = 9.81
    mu_c: float = 1.0
    mu_p: float = 1.1
    gamma: float = 100.0
