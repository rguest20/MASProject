from evolution.coordinator import Coordinator
from config import GENERATIONS

coord = Coordinator()

for _ in range(GENERATIONS):
    coord.run_generation()
