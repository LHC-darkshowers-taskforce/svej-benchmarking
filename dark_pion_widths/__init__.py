from .base import DarkPionModelBase, HBAR_C_GEV_MM
from .generators import (
    build_sun_generators,
    classify_generator,
    get_diagonal_indices,
    get_diagonal_names,
)
from .tchannel import DarkPionTChannelModel, DEFAULT_QUARKS
from .schannel import DarkPionSChannelModel, DEFAULT_SM_QUARKS, loop_function_squared

# Updated constants for Nf = 4 (new generator ordering)
# T3 → index 12,  T8 → index 13,  T15 → index 14
DIAGONAL_PION_INDICES: list[int]      = get_diagonal_indices(4)   # [12, 13, 14]
DIAGONAL_PION_NAMES:   dict[int, str] = get_diagonal_names(4)     # {12:'T3', 13:'T8', 14:'T15'}
