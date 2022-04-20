from modular_robots_2d import REM2D_main as r2d
from modular_robots_2d import Tree
from Controllers.Controller import Controller
import copy
import random

class BinaryEncoding():
    def __init__(self, module_list):
        # This list contains the modules to choose from. Needs to be deep-copied when used
        super().__init__()
        self.module_list = module_list

        self.control_list = []
        for i in range(len(module_list)):
            self.module_list[i].mutate(0.5, 0.5)
            controller = Controller()
            self.control_list.append(controller)
        
        self.tree = None

        self.n_modules = 1
        self.maxDepth = 8
        self.maxModules = 20

        self.genome_length = 112 
        self.genome_part_length = 6
        self.create_module = self.create_module_6
        self.create_module_list = self.create_module_list_176
        self.module_list_length = 176
        self.genome = [random.randint(0,1) for x in range(self.module_list_length+self.genome_length)]

    """
    Genome structure: 
      - Exists: No/Yes (1 bit)
      - If exists:
        - Type: Simple/Circle (1 bit)
        - Module params:
          - Width: (5 bit)
          - Height: (5 bit)
          - Angle: (5 bit)
        - Controller:
          - Amplitude: (5 bit)
          - Period: (5 bit)
          - Phase: (5 bit)
    """
        
    def create(self):
        self.tree = Tree.Tree(self.module_list)

        self.genome[self.module_list_length] = 0

        self.create_module_list(self.genome[0:self.module_list_length])
        # Parent module
        self.create_module(self.genome[0+self.module_list_length:self.genome_part_length+self.module_list_length], self.tree, 0, -1, None)

        node_index = 0

        node_counter = 1
        connection_counter = 0
        
        # The rest
        genome_index = self.genome_part_length + self.module_list_length #31
        while(genome_index < len(self.genome)-self.genome_part_length and node_index < len(self.tree.nodes)):
            availableConnections = self.tree.nodes[node_index].module_.available
            #print(availableConnections)
            if(len(availableConnections) > 0):
                if(self.genome[genome_index] == 1):
                    self.create_module(self.genome[1+genome_index:self.genome_part_length+1+genome_index], self.tree, node_counter, node_index, availableConnections[connection_counter])
                    node_counter += 1
                    genome_index += self.genome_part_length + 1 #32
                    connection_counter += 1
                else:
                    genome_index += 1
                    connection_counter += 1
            if connection_counter >= len(availableConnections):
                connection_counter = 0
                node_index += 1
        return self.tree

    def get_num_modules(self):
        self.tree = Tree.Tree(self.module_list)

        self.genome[self.module_list_length] = 0

        self.create_module_list(self.genome[0:self.module_list_length])
        # Parent module
        self.create_module(self.genome[0+self.module_list_length:self.genome_part_length+self.module_list_length], self.tree, 0, -1, None)

        node_index = 0

        node_counter = 1
        connection_counter = 0
        
        # The rest
        genome_index = self.genome_part_length + self.module_list_length #31
        while(genome_index < len(self.genome)-self.genome_part_length and node_index < len(self.tree.nodes)):
            availableConnections = self.tree.nodes[node_index].module_.available
            #print(availableConnections)
            if(len(availableConnections) > 0):
                if(self.genome[genome_index] == 1):
                    self.create_module(self.genome[1+genome_index:self.genome_part_length+1+genome_index], self.tree, node_counter, node_index, availableConnections[connection_counter])
                    node_counter += 1
                    genome_index += self.genome_part_length + 1 #32
                    connection_counter += 1
                else:
                    genome_index += 1
                    connection_counter += 1
            if connection_counter >= len(availableConnections):
                connection_counter = 0
                node_index += 1
        #return node_counter
        return ratio

    def create_module_list_176(self, genome):
        for i in range(8):
            if i < 4:
                width = self.n_bit_to_int(genome[0+i*24:4+i*24], 4)/15.0
                height = self.n_bit_to_int(genome[4+i*24:8+i*24], 4)/15.0
                angle = self.n_bit_to_int(genome[8+i*24:12+i*24], 4)/15.0
        
                amplitude = self.n_bit_to_int(genome[12+i*24:16+i*24], 4)/15.0
                period = self.n_bit_to_int(genome[16+i*24:20+i*24], 4)/15.0
                phase = self.n_bit_to_int(genome[20+i*24:24+i*24], 4)/15.0
            else:
                width = self.n_bit_to_int(genome[96+((i-4)*20):4+96+((i-4)*20)], 4)/15.0
                height = 0.0
                angle = self.n_bit_to_int(genome[4+96+((i-4)*20):8+96+((i-4)*20)], 4)/15.0
        
                amplitude = self.n_bit_to_int(genome[8+96+((i-4)*20):12+96+((i-4)*20)], 4)/15.0
                period = self.n_bit_to_int(genome[12+96+((i-4)*20):16+96+((i-4)*20)], 4)/15.0
                phase = self.n_bit_to_int(genome[16+96+((i-4)*20):20+96+((i-4)*20)], 4)/15.0
            self.module_list[i].setMorph(width, height, angle)
            self.control_list[i] = Controller(amplitude, period, phase)

    def create_module_list_220(self, genome):
        for i in range(8):
            if i < 4:
                width = self.five_bit_to_int(genome[0+i*30:5+i*30])/31.0
                height = self.five_bit_to_int(genome[5+i*30:10+i*30])/31.0
                angle = self.five_bit_to_int(genome[10+i*30:15+i*30])/31.0
        
                amplitude = self.five_bit_to_int(genome[15+i*30:20+i*30])/31.0
                period = self.five_bit_to_int(genome[20+i*30:25+i*30])/31.0
                phase = self.five_bit_to_int(genome[25+i*30:30+i*30])/31.0
            else:
                width = self.five_bit_to_int(genome[0+120+((i-4)*25):5+120+((i-4)*25)])/31.0
                height = 0.0
                angle = self.five_bit_to_int(genome[0+120+((i-4)*25):10+120+((i-4)*25)])/31.0
        
                amplitude = self.five_bit_to_int(genome[10+120+((i-4)*25):15+120+((i-4)*25)])/31.0
                period = self.five_bit_to_int(genome[15+120+((i-4)*25):20+120+((i-4)*25)])/31.0
                phase = self.five_bit_to_int(genome[20+120+((i-4)*25):25+120+((i-4)*25)])/31.0
            self.module_list[i].setMorph(width, height, angle)
            self.control_list[i] = Controller(amplitude, period, phase)
            

    def create_module_31(self, genome31, tree, index, parent, connection_site):
        if genome31[0] == 0:
            module = copy.deepcopy(self.module_list[0])
        else:
            module = copy.deepcopy(self.module_list[4])

        width = self.five_bit_to_int(genome31[1:6])/31.0
        height = self.five_bit_to_int(genome31[6:11])/31.0
        angle = self.five_bit_to_int(genome31[11:16])/31.0
        
        amplitude = self.five_bit_to_int(genome31[16:21])/31.0
        period = self.five_bit_to_int(genome31[21:26])/31.0
        phase = self.five_bit_to_int(genome31[26:31])/31.0

        control = Controller(amplitude, period, phase)
        module.setMorph(width, height, angle)
        
        tree.nodes.append(Tree.Node(index, parent, 0, connection_site, control, module, module_=module))

    def create_module_6(self, genome_part, tree, index, parent, connection_site):
        module_index = self.n_bit_to_int(genome_part[0:3], 3)
        control_index = self.n_bit_to_int(genome_part[3:6], 3)

        module = copy.deepcopy(self.module_list[module_index])
        control = copy.deepcopy(self.control_list[control_index])
        
        tree.nodes.append(Tree.Node(index, parent, module_index, connection_site, control, module, module_=module))

    def five_bit_to_int(self, part_5_bit):
        result = 0
        for i in range(5):
            result += part_5_bit[i]*(2**i)
        return float(result)

    def n_bit_to_int(self, part_n_bit, n):
        result = 0
        for i in range(n):
            result += part_n_bit[-1-i]*(2**i)
        return result

    def compare(self, other):
        diff = 0
        for i in range(len(self.genome)):
            if (self.genome[i] != other.genome[i]):
                diff += 1
        return diff
            
    def mutate(self, mutation_rate):
        for i in range(len(self.genome)):
            if random.uniform(0,1) < mutation_rate:
                self.genome[i] = random.randint(0, 1)

    def mutate_one_bit(self):
        i = random.randint(0,len(self.genome)-1)
        if self.genome[i] == 0:
            self.genome[i] = 1
        else:
            self.genome[i] = 0

    def create_all_one_bit_children(self):
        children = []
        for i in range(len(self.genome)):
            child = copy.deepcopy(self)
            if child.genome[i] == 0:
                child.genome[i] = 1
            else:
                child.genome[i] = 0
            children.append(child)
        return children

    def create_one_bit_child(self, i):
        child = copy.deepcopy(self)
        if child.genome[i] == 0:
            child.genome[i] = 1
        else:
            child.genome[i] = 0
        return child
    
    def crossover(self, other):
        point = random.randint(0,len(self.genome)-1)
        flip = random.randint(0, 1)
        if flip:
            for i in range(len(self.genome)):
                if i < point:
                    self.genome[i] = other.genome[i]
        else:
            for i in range(len(self.genome)):
                if i > point:
                    self.genome[i] = other.genome[i]

    def uniform_crossover(self, other):
        for i in range(len(self.genome)):
            if random.randint(0, 1) == 1:
                tmp = self.genome[i]
                self.genome[i] = other.genome[i]
                other.genome[i] = tmp

    def n_point_crossover(self, other, n):
        points = [random.randint(0,len(self.genome)-1) for _ in range(n)]

        flip = random.randint(0, 1)
        for i in range(len(self.genome)):
            if flip == 1:
                tmp = self.genome[i]
                self.genome[i] = other.genome[i]
                other.genome[i] = tmp
            if i in points:
                if flip == 0:
                    flip = 1
                else:
                    flip = 0

    def make_complement(self, other):
        for i in range(len(self.genome)):
            if other.genome[i] == 0:
                self.genome[i] = 1
            else:
                self.genome[i] = 0

    def bit_edit_distance(self, other):
        dist = 0
        for i in range(len(self.genome)):
            if other.genome[i] != self.genome[i]:
                dist += 1
        return dist
