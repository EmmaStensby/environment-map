import copy
import numpy as np
import neat
#import matplotlib.pyplot as plt

class Environment:
    def __init__(self, seed, mut_rate, env_length, config=None, genome_config=None, genome=None):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.genome_config = genome_config
        self.config = config
        
        self.mut_rate = mut_rate
        self.env_length = env_length
        
        #self.env_seed = int(self.rng.integers(0, 2**63-2))

        self.genome = genome

    def get_copy(self):
        c1 = Environment(self.seed.spawn(1)[0], self.mut_rate, self.env_length, self.config, self.genome_config, copy.deepcopy(self.genome))
        return c1

    def mutate(self):
        if self.rng.uniform(0,1) < self.mut_rate:
            self.genome.mutate(self.genome_config)

    def guaranteed_mutate(self):
        self.genome.mutate(self.genome_config)
            
    def get_nodes(self):
        nodes, _ = self.genome.size()
        return nodes
        
    def get_terrain(self, max_perturbance):
        terrain = []
        nn = neat.nn.feed_forward.FeedForwardNetwork.create(self.genome, self.config)
        for i in range(self.env_length):
            terrain.append(nn.activate([i/self.env_length])[0])
        for i in range(len(terrain)):
            if terrain[i] < 0.0:
                terrain[i] = 0.0
            if terrain[i] > 24.0:
                terrain[i] = 24.0
        return terrain

    def show(self):
        #terrain = self.get_terrain(24.0)
        #plt.plot(terrain)
        #plt.show()
        pass

    def compare(self, other):
        a = self.get_normalised_terrain()
        b = other.get_normalised_terrain()
        diff = 0
        for x, y in zip(a, b):
            diff += abs(x - y)
        return diff
    
    def get_normalised_terrain(self):
        terrain = self.get_terrain(None)
        for i in range(len(terrain)):
            terrain[i] = (terrain[i])/24
        return np.array(terrain)

    def get_frechet_terrain(self):
        terrain = self.get_terrain(None)
        for i in range(len(terrain)):
            terrain[i] = [(terrain[i])/24]
        return terrain

    def is_flat_terrain(self):
        terrain = self.get_terrain(None)
        for i in range(len(terrain)):
            if terrain[i] > 0.01 or terrain[i] < -0.01:
                return False
        return True
