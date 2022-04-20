import pickle
import numpy as np
import multiprocessing as mp
from multiprocessing import pool

from Individuals.modular_2d import Individual
from Environments.terrain import Environment
from Evaluators.modular_2d_evaluator import Evaluator
from Utilities.autoencoder import Autoencoder

from modular_robots_2d import REM2D_main as r2d

import neat
import copy

class MapEnv:
    def __init__(self, seed, run_id, resolution, params=None):


        # --- Seed ---
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        
        # --- ID ---
        self.run_id = run_id

        # --- Map ---
        self.params = params
        self.resolution = resolution
        self.static_resolution = 10

        # Env length
        env = r2d.get_env()
        self.env_length = env.get_terrain_length()
        
        # --- Static map
        self.static_solution_map = None
        self.static_fitness_map = None
        self.init_static_maps(self.static_resolution)

         # --- Dynamic map
        self.dynamic_solution_map = None
        self.dynamic_fitness_map = None
        self.init_dynamic_maps(self.resolution)
        self.ae_1 = Autoencoder(self.env_length)
        self.ae_train_period = params[5]
        
        # --- Params ---
        self.morph_mut_rate = params[0]
        self.env_mut_rate = params[1]

        self.init_generation_size = params[2]
        self.generation_size = params[2]

        self.insert_fitness_criterion = params[3]
        self.mutate_env_fitness_criterion = params[4]

        # --- Statistics ---
        self.added_over_time = []
        self.static_added_over_time = []
        
        self.map_fullness = 0
        self.static_map_fullness = 0
        self.map_fullness_over_time = []
        self.static_map_fullness_over_time = []
        
        self.max_fitness_over_time = []
        self.mean_fitness_over_time = []
        self.static_mean_fitness_over_time = []

        self.solved_environment_archive = []
        self.solved_environment_times = []
        self.solved_archive_threshold = 2.5
        self.solved_archive_fitness_threshold = 200

        self.found_environment_archive = []
        self.found_environment_times = []
        self.found_archive_threshold = 25 #2.5
        self.found_archive_fitness_threshold = 0

        self.num_mutations = 10 # 100

        self.current_evaluations = 0
        
        # --- Other ---
        # Threads
        self.threads = 16 #mp.cpu_count()
        print("Threads: ", self.threads)

        # Iteration counter
        self.total_iterations = 0

        # Set up evaluator
        self.evaluator = Evaluator()

        # Checkpoints
        self.checkpoint_period = 1

        # Hackityhack
        config_file = "neat-cppn-config-poet"
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
        self.p = neat.Population(self.config)

    def init_dynamic_maps(self, resolution):
        self.dynamic_fitness_map = [[np.nan for b in range(resolution*resolution)] for a in range(resolution*resolution)]
        self.dynamic_solution_map = [[None for b in range(resolution*resolution)] for a in range(resolution*resolution)]

    def init_static_maps(self, resolution):
        self.static_fitness_map = [[np.nan for b in range(resolution*resolution)] for a in range(resolution*resolution)]
        self.static_solution_map = [[None for b in range(resolution*resolution)] for a in range(resolution*resolution)]

    def run(self, iterations):
        for i in range(iterations):
            print("---------------")
            print("Iteration: ", self.total_iterations+1)
            print("---------------")
            
            map_added = 0
            static_map_added = 0
            
            # Run
            if self.total_iterations == 0:
                initial_environments = []
                for x in range(self.init_generation_size):
                    initial_environments.append(Environment(self.seed.spawn(1)[0], self.env_mut_rate, self.env_length, config=self.config, genome_config=self.config.genome_config, genome=self.p.population[x+1]))
                

                genomes = self.get_ae_data_init(initial_environments)
                self.ae_1 = Autoencoder(self.env_length)
                self.ae_1.train(np.array(genomes), 7500)
                    
                for i in range(len(initial_environments)):
                    for j in range(self.num_mutations):
                        initial_environments[i].guaranteed_mutate()
                    
                pairs = [[Individual(self.seed.spawn(1)[0], self.morph_mut_rate), initial_environments[x]] for x in range(self.init_generation_size)]
                fitness_list = self.evaluate_parallel(pairs)
                self.current_evaluations += self.init_generation_size
                for pair, fitness in zip(pairs, fitness_list):
                    if(self.place_in_map(pair, fitness, initial=True)):
                        map_added += 1
                    if self.place_in_static_map(pair, fitness):
                        static_map_added += 1
                    if fitness > self.solved_archive_fitness_threshold:
                        self.add_to_solved_environment_archive(pair)
                    if fitness > self.found_archive_fitness_threshold:
                        self.add_to_found_environment_archive(pair)

            else:
                parents_and_fitnesses_1 = [self.select() for x in range(self.generation_size)]
                parents_and_fitnesses_2 = [self.select() for x in range(self.generation_size)]
                
                children = []
                
                for p_1, p_2 in zip(parents_and_fitnesses_1, parents_and_fitnesses_2):
                    parent_1 = p_1[0]
                    fitness_1 = p_1[1]
                    new_1 = self.mutation(parent_1, fitness_1)
                    children.append(new_1)
                
                fitness_list = self.evaluate_parallel(children)
                self.current_evaluations += self.generation_size
                for pair, fitness in zip(children, fitness_list):
                    if(self.place_in_map(pair, fitness)):
                        map_added += 1
                    if self.place_in_static_map(pair, fitness):
                        static_map_added += 1
                    if fitness > self.solved_archive_fitness_threshold:
                        self.add_to_solved_environment_archive(pair)
                    if fitness > self.found_archive_fitness_threshold:
                        self.add_to_found_environment_archive(pair)
            self.total_iterations += 1

            # Status
            self.added_over_time.append(map_added)
            self.static_added_over_time.append(static_map_added)
            self.map_fullness_over_time.append(self.map_fullness)
            self.static_map_fullness_over_time.append(self.static_map_fullness)
            self.max_fitness_over_time.append(np.nanmax(self.dynamic_fitness_map))
            self.mean_fitness_over_time.append(np.nanmean(self.dynamic_fitness_map))
            self.static_mean_fitness_over_time.append(np.nanmean(self.static_fitness_map))
            
            print("Map fullness: ", self.map_fullness,"/", self.resolution**4)
            print("Max fitness: ", self.max_fitness_over_time[-1])
            print("Mean fitness: ", self.mean_fitness_over_time[-1])
            print("Added: ", map_added,"/{}".format(self.generation_size))
            print("Found archive: ", len(self.found_environment_archive))

            # Retrain AE
            if self.total_iterations % self.ae_train_period == 0 and len(self.solved_environment_archive) > 0:
                genomes = self.get_ae_data()
                self.ae_1 = Autoencoder(self.env_length)
                self.ae_1.train(np.array(genomes), 7500)
                old_solutions = []
                old_fitnesses = []
                for a in range(self.resolution*self.resolution):
                    for b in range(self.resolution*self.resolution):
                        if self.dynamic_solution_map[a][b] is not None:
                            old_solutions.append(self.dynamic_solution_map[a][b])
                            old_fitnesses.append(self.dynamic_fitness_map[a][b])
                self.dynamic_solution_map = [[None for b in range(self.resolution*self.resolution)] for a in range(self.resolution*self.resolution)]
                self.dynamic_fitness_map = [[np.nan for b in range(self.resolution*self.resolution)] for a in range(self.resolution*self.resolution)]
                self.map_fullness = 0
                self.logg_to_file([-1,-1], -1, "checkpoints/logg_{}.txt".format(self.run_id))
                map_added = 0
                for pair, fitness in zip(old_solutions, old_fitnesses):
                    if(self.place_in_map(pair, fitness)):
                        map_added += 1
                print("*Retrained ae:")
                print("Map fullness: ", self.map_fullness,"/", self.resolution**4)
            
 
            # Checkpoint
            if True: #self.total_iterations % self.checkpoint_period == 0:
                path = "checkpoints/checkpoint_{}.pkl".format(self.run_id)
                
                self.ae_1.save_self("checkpoints/autoencoder_{}".format(self.run_id))

                with open(path, "wb") as f:
                    save_info = {}
                    
                    save_info["fitness_d"] = self.dynamic_fitness_map
                    save_info["params"] = self.params
                    save_info["res"] = self.resolution
                    save_info["individuals_d"] = self.dynamic_solution_map
                    save_info["fitness_s"] = self.static_fitness_map
                    save_info["individuals_s"] = self.static_solution_map
                    save_info["individuals_a"] = self.solved_environment_archive
                    save_info["individuals_f"] = self.found_environment_archive
                    save_info["individuals_at"] = self.solved_environment_times
                    save_info["individuals_ft"] = self.found_environment_times
                    save_info["added_over_time"] = self.added_over_time
                    save_info["static_added_over_time"] = self.static_added_over_time
                    save_info["map_fullness_over_time"] = self.map_fullness_over_time
                    save_info["max_fitness_over_time"] = self.max_fitness_over_time
                    save_info["mean_fitness_over_time"] = self.mean_fitness_over_time
                    save_info["static_map_fullness_over_time"] = self.static_map_fullness_over_time
                    save_info["static_mean_fitness_over_time"] = self.static_mean_fitness_over_time
                    pickle.dump(save_info, f)
            

    def add_to_solved_environment_archive(self, pair):
        if len(self.solved_environment_archive) == 0:
            env = pair[1]
            ind = pair[0]
            self.solved_environment_archive.append([ind, env])
            self.solved_environment_times.append(self.current_evaluations)
        else:
            env = pair[1]
            ind = pair[0]
            differences = []
            for a_pair in self.solved_environment_archive:
                differences.append(env.compare(a_pair[1]))
            if min(differences) > self.solved_archive_threshold:
                self.solved_environment_archive.append([ind, env])
                self.solved_environment_times.append(self.current_evaluations)

    def add_to_found_environment_archive(self, pair):
        if len(self.found_environment_archive) == 0:
            env = pair[1]
            ind = pair[0]
            self.found_environment_archive.append([ind, env])
            self.found_environment_times.append(self.current_evaluations)
        else:
            env = pair[1]
            ind = pair[0]
            differences = []
            for a_pair in self.found_environment_archive:
                differences.append(env.compare(a_pair[1]))
            if min(differences) > self.found_archive_threshold:
                self.found_environment_archive.append([ind, env])
                self.found_environment_times.append(self.current_evaluations)

    def place_in_map(self, pair, fitness, initial=False):
        a, b = self.identify_map_position(pair[1])
        if self.dynamic_fitness_map[a][b] is np.nan:
            # Add to map
            if (fitness > self.insert_fitness_criterion) or initial:
                self.dynamic_solution_map[a][b] = pair
                self.dynamic_fitness_map[a][b] = fitness
                self.logg_to_file([a,b], fitness, "checkpoints/logg_{}.txt".format(self.run_id))
                self.map_fullness += 1
                return True
        elif fitness > self.dynamic_fitness_map[a][b]:
            # Replace in map
            self.dynamic_solution_map[a][b] = pair
            self.dynamic_fitness_map[a][b] = fitness
            self.logg_to_file([a,b], fitness, "checkpoints/logg_{}.txt".format(self.run_id))
            return True
        return False

    def place_in_static_map(self, pair, fitness, initial=False):
        a, b = self.identify_static_map_position(pair[1])
        if self.static_fitness_map[a][b] is np.nan:
            self.static_solution_map[a][b] = pair
            self.static_fitness_map[a][b] = fitness
            self.logg_to_file([a,b], fitness, "checkpoints/logg_static_{}.txt".format(self.run_id))
            self.static_map_fullness += 1
            return True
        elif fitness > self.static_fitness_map[a][b]:
            # Replace in map
            self.static_solution_map[a][b] = pair
            self.static_fitness_map[a][b] = fitness
            self.logg_to_file([a,b], fitness, "checkpoints/logg_static_{}.txt".format(self.run_id))
            return True
        return False

    def logg_to_file(self, coords, fitness, path):
        with open(path, "a") as f:
            for c in coords:
                f.write(str(c) + " ")
            f.write(str(fitness) + " ")
            f.write(str(self.current_evaluations) + " ")
            f.write("\n")

    def identify_map_position(self, env):
        genome = env.get_normalised_terrain()
        genome_processed = self.ae_1.process(np.array([genome]))
        
        d1 = 0
        for x, y in zip(genome, genome_processed[0]):
            d1 += np.abs(x-y)/len(genome)

        d1 = d1*5
            
        d2 = env.get_nodes()
        
        # Find map coordinates
        a = 0
        for i in range((self.resolution*self.resolution)-1):
            if d1 > (1/(self.resolution*self.resolution))*(i+1):
                a += 1
            else:
                break
        b = 0
        for i in range((self.resolution*self.resolution)-1):
            if d2 > i+1:
                b += 1
            else:
                break
        return a, b

    def identify_static_map_position_2(self, env):
        genome = env.get_normalised_terrain()
        
        d1 = 0
        for i in range(len(genome)-1):
            d1 += np.abs(genome[i+1] - genome[i])/(len(genome)-1)

        d1 = d1*50
            
        d2 = max(genome)
        
        # Find map coordinates
        a = 0
        for i in range((self.resolution*self.resolution)-1):
            if d1 > (1/(self.resolution*self.resolution))*(i+1):
                a += 1
            else:
                break
        b = 0
        for i in range((self.resolution*self.resolution)-1):
            if d2 > (1/(self.resolution*self.resolution))*(i+1):
                b += 1
            else:
                break
        return a, b

    def identify_static_map_position(self, env):
        genome = env.get_normalised_terrain()

        d1 = 0
        for i in range(len(genome)-1):
            d1 += np.abs(genome[i+1] - genome[i])/(len(genome)-1)

        d1 = d1*50

        d2 = max(genome)

        # Find map coordinates                                                                                                                                                                              
        a = 0
        for i in range((self.static_resolution*self.static_resolution)-1):
            if d1 > (1/(self.static_resolution*self.static_resolution))*(i+1):
                a += 1
            else:
                break
        b = 0
        for i in range((self.static_resolution*self.static_resolution)-1):
            if d2 > (1/(self.static_resolution*self.static_resolution))*(i+1):
                b += 1
            else:
                break
        return a, b


    def get_ae_data(self):
        genomes = []
        for a in range(self.resolution*self.resolution):
            for b in range(self.resolution*self.resolution):
                if self.dynamic_solution_map[a][b] is not None:
                    env = self.dynamic_solution_map[a][b][1]
                    genome = env.get_normalised_terrain()
                    genomes.append(genome)
        return genomes
    
    def get_ae_data_init(self, initial_environments):
        genomes = []
        for env in initial_environments:
            genome = env.get_normalised_terrain()
            genomes.append(genome)
        return genomes
    
    
    def get_ae_data_archive(self):
        genomes = []
        for pair in self.solved_environment_archive:
            env = pair[1]
            genome = env.get_normalised_terrain()
            genomes.append(genome)
        return genomes

    def select(self):
        while(True):
            a = self.rng.integers(self.resolution*self.resolution)
            b = self.rng.integers(self.resolution*self.resolution)
            if self.dynamic_solution_map[a][b] is not None:
                return [self.dynamic_solution_map[a][b], self.dynamic_fitness_map[a][b]]

    def mutation(self, pair, fitness):
        i1 = pair[0].get_copy()
        e1 = pair[1].get_copy()
        i1.mutate()
        e1.mutate()
        new_1 = [i1, e1]
        return new_1

    def evaluate_parallel(self, pairs):
        parameters = []
        for i in range(len(pairs)):
            parameters.append((pairs[i][0], pairs[i][1]))
        with pool.Pool(self.threads) as p:
            fitness_list = p.starmap(self.evaluator.evaluate, parameters)
        return fitness_list
    
    
def run():

    seed = np.random.SeedSequence()
    rng = np.random.default_rng(seed)
    run_id= int(rng.integers(0, 2**63-2))
    map_res = 5 #rng.choice([5, 8, 10])
    
    # Select random hyperparameters
    morph_mut_rate = rng.choice([0.05, 0.1, 0.2])
    env_mut_rate = rng.choice([0.05, 0.1, 0.2])
    
    generation_size = 500 #rng.choice([50, 100, 500])

    insert_fitness_criterion = rng.choice([50, 100, 150])
    mutate_env_fitness_criterion = rng.choice([50, 100, 150])
    
    ae_train_period = rng.choice([20, 50, 100])
    
    params=[morph_mut_rate, env_mut_rate, generation_size, insert_fitness_criterion, mutate_env_fitness_criterion, ae_train_period]
    #params=[0.05, 0.1, 500, 100, 200, 100]
    params=[0.05, 0.2, 500, 100, 50, 100] #0] #100]

    print("Run id: ", run_id)
    print("Params: ", params)
    print("Seed: ", seed)
    print("Resolution: ", map_res)

    # Run
    map_env = MapEnv(seed.spawn(1)[0], run_id, map_res, params=params)
    map_env.run(100000)

