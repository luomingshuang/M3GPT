from .solver import Solver
from .solver_deter import SolverDeter
from .solver_multitask_dev import SolverMultiTaskDev

from .solver_multitask_dev import TesterMultiTaskDev

def solver_entry(C):
    # print(C.config)
    return globals()[C.config['common']['solver']['type']](C)
