from .curriculum import (
    ZONES_CURRICULUM, LETTER_CURRICULUM, FLATWORLD_CURRICULUM, FLATWORLD_BIG_CURRICULUM,
    ZONES_PLANNING_CURRICULUM, HARD_OPTIMALITY_CURRICULUM,
)
from .curriculum_sampler import CurriculumSampler

curricula = {
    'PointLtl2-v0': ZONES_CURRICULUM,
    'PointLtl2-planning-v0': ZONES_PLANNING_CURRICULUM,  # Extended with planning stages
    'PointLtl2-v0.hardmix': HARD_OPTIMALITY_CURRICULUM,  # Hard optimality training
    'LetterEnv-v0': LETTER_CURRICULUM,
    'FlatWorld-v0': FLATWORLD_CURRICULUM,
    'FlatWorld-big-v0': FLATWORLD_BIG_CURRICULUM,
}
