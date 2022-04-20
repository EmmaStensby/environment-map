from modular_robots_2d import REM2D_main as r2d
import gym
import imageio as io
gym.logger.set_level(40)

class Evaluator:
    def __init__(self, render_interval=1):
        self.max_steps = 2000
        self.stability_of_evaluation = 1 ### Fix stuff if change this.Is not used in evaluate right now!!!!!
        self.render_interval = render_interval

    def evaluate(self, encoding, environment, render=False):
        # Set up robot
        tree = encoding.individual.create()
    
        # Set up environment
        env = r2d.get_env()
        module_list = encoding.module_list
        environment_length = env.get_terrain_length()
        env.seed(4)
        env.reset(tree=tree, module_list=tree.module_list, terrain=environment.get_terrain(env.get_max_perturbance_terrain()))
    
        # Evaluation parameters
        max_steps = 2000

        # Evaluation
        fitness = 0
        for i in range(max_steps):
            if i % self.render_interval == 0:
                if render:
                    env.render()      
            observation, reward, done, info  = env.step(i)              
            if reward < -10: 
                break
            elif reward > environment_length:
                fitness = reward
                break
            if reward > 0:
                fitness = reward       
        if render:
            env.close()

        return fitness

    
    def save_video(self, encoding, environment):
        # Set up robot
        tree = encoding.individual.create()
    
        # Set up environment
        env = r2d.get_env()
        module_list = encoding.module_list
        environment_length = env.get_terrain_length()
        env.seed(4)
        env.reset(tree=tree, module_list=tree.module_list, terrain=environment.get_terrain(env.get_max_perturbance_terrain()))
    
        # Evaluation parameters
        max_steps = 2000

        # Evaluation
        fitness = 0
        gif = []
        for i in range(max_steps):
            if i % self.render_interval == 0:
                img = env.render('rgb_array')
                gif.append(img)
            observation, reward, done, info  = env.step(i)              
            if reward < -10: 
                break
            elif reward > environment_length:
                fitness = reward
                break
            if reward > 0:
                fitness = reward       
        env.close()

        io.mimsave("example_video_1.gif", gif, fps=60)

        return fitness

    def get_image(self, encoding, environment):
        # Set up robot
        tree = encoding.individual.create()
    
        # Set up environment
        env = r2d.get_env()
        module_list = encoding.module_list
        environment_length = env.get_terrain_length()
        env.seed(4)
        env.reset(tree=tree, module_list=tree.module_list, terrain=environment.get_terrain(env.get_max_perturbance_terrain()))
    
        # Evaluation parameters
        max_steps = 1

        # Evaluation
        fitness = 0
        gif = []
        for i in range(max_steps):
            img = env.render('rgb_array')
            gif.append(img)
            observation, reward, done, info  = env.step(i)              
            if reward < -10: 
                break
            elif reward > environment_length:
                fitness = reward
                break
            if reward > 0:
                fitness = reward       
        env.close()

        return gif[0]
