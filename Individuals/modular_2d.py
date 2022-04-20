from modular_robots_2d import REM2D_main as r2d
from Encodings.BinaryEncoding import BinaryEncoding
import copy

class Individual:
    def __init__(self, seed, mut_rate):
        self.seed = seed
        
        self.morph_mut_rate = mut_rate
        
        self.module_list = r2d.get_module_list()

        self.individual = BinaryEncoding(self.module_list)

    def get_copy(self):
        c1 = Individual(self.seed.spawn(1)[0], self.morph_mut_rate)
        c1.individual = copy.deepcopy(self.individual)
        return c1

    def mutate(self):
        self.individual.mutate(self.morph_mut_rate)

    def get_num_modules(self):
        return self.individual.get_num_modules()
