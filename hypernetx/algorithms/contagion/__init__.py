from hypernetx.algorithms.contagion.animation import contagion_animation
from hypernetx.algorithms.contagion.epidemics import (collective_contagion, individual_contagion, threshold,
                                                      majority_vote, discrete_SIR, discrete_SIS, Gillespie_SIR, Gillespie_SIS)

__all__ = ["contagion_animation",
    "collective_contagion",
    "individual_contagion",
    "threshold",
    "majority_vote",
    "discrete_SIR",
    "discrete_SIS",
    "Gillespie_SIR",
    "Gillespie_SIS",]