from evolution.coordinator import Coordinator
from config import GENERATIONS
import cProfile

coord = Coordinator()

for _ in range(GENERATIONS):
    coord.run_generation()
    # cProfile.run('coord.run_generation()')
