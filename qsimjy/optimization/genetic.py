import random
import numpy as np
import copy
from .. import magnet

#Scientific Constants
h = 4.13e-15     # eV*s
guB = 2 * 57.8e-6 # eV / T

class ChromParams():
    def __init__(self,
                 mesh_size : [int, int],
                 mesh_x_list: [float, float],
                 mesh_y_list: [float, float],
                 MmSimParams: magnet.MmSimParams,
                 z_gran: int,
                 n_cpu : int = 4
                 ):
        self.mesh_size = mesh_size
        self.mesh_x_list = mesh_x_list
        self.mesh_y_list = mesh_y_list
        self.MmSimParams = MmSimParams
        self.z_gran = z_gran
        self.n_cpu = n_cpu
        
class StrayCalcParams():
    def __init__(self,
                 x_range : [float, float],
                 y_range : [float, float],
                 gran : [int, int],
                 quantum_dot_pos_z : float):
        self.field_x_range = x_range
        self.field_y_range = y_range
        self.gran = gran
        self.qunatum_dot_pos_z = quantum_dot_pos_z
        
class ExpParams():
    def __init__(self,
                 osc_pos : float, # amplitude of the QD position of the Rabi driving
                 sigma_pos : float, # noise in the QD position
                 qd_positions : list): # quantum dot positions to calculate
        self.osc_pos = osc_pos
        self.sigma_pos = sigma_pos
        self.qd_positions = qd_positions


def RandomMeshGen(mesh_size): 
    """
    Creates a random binary mesh with the dimension of `mesh_size`.
    """
    return np.random.randint(0, 2, size = mesh_size) 

    
def Mutate(chrom,
           mutation_rate = 0.001,#default mutation rate is 0.00: 1
           mut_constraint = None):
    """
    Mutates a chromosome. `mut_constaint` chooses one fo the following mutation consraints:
    1. None: completely random mutation
    2. 'adj' = mutation is allowed if the adjacent cell has the same value to mutate
    """
    if mut_constraint == None:
        for i in range(np.shape(chrom)[0]):
            for j in range(np.shape(chrom)[1]):
                if random.random() < mutation_rate:
                    chrom[i,j] = 1-chrom[i,j]
        return 0
    elif mut_constraint == 'adj':
        _n_x,_n_y =  np.shape(chrom)[0],np.shape(chrom)[1]
        _mutated = copy.deepcopy(chrom)
        _mutation_mask = np.random.rand(_n_x, _n_y) < mutation_rate #Boolean mask of a cell to mutate
        _padded = np.pad(_mutated, pad_width = 1, mode = 'reflect')
        _neighbor_ones = (_padded[:-2, 1:-1] +  
                          _padded[2:, 1:-1] +   
                          _padded[1:-1, :-2] +  
                          _padded[1:-1, 2:]) > 0
        _neighbor_zeros = ((_padded[:-2, 1:-1] == 0).astype(int) + 
                           (_padded[2:, 1:-1] == 0).astype(int) +  
                           (_padded[1:-1, :-2] == 0).astype(int) +  
                           (_padded[1:-1, 2:] == 0).astype(int)) > 0
        _can_mutate_01 = (_mutated == 0) & _neighbor_ones #Boolean mask for the constraint 0 to 1
        _can_mutate_10 = (_mutated == 1) & _neighbor_zeros #Boolean mask for the constraint 1 to 0
        _mutated[(_mutation_mask & _can_mutate_01)] = 1  # Mutation: 0 → 1 
        _mutated[(_mutation_mask & _can_mutate_10)] = 0  # Mutation: 1 → 0
        return _mutated


def InitChromGen(chrom_params: ChromParams,
                 population_size : int,
                 is_random = True,
                 init_input: magnet.MagnetMesh = None,
                 mutation_rate = 0.001):
    """
    Generates a list of randomly prepared initial chromosomes, which is a MagnetMesh obj.
    `mesh_size` indicates the size of a mesh for each chromosome.
    `x_list` and `y_list` set the boundary for the mesh. 
    """
    if is_random != True:
        _to_return = copy.deepcopy(init_input.mesh_coord)
        Mutate(_to_return, 
               mutation_rate, 
               'adj')
        return [magnet.MagnetMesh(_to_return,
                                  chrom_params.mesh_x_list,
                                  chrom_params.mesh_y_list) for _ in range(population_size)]
    return [magnet.MagnetMesh(RandomMeshGen(chrom_params.mesh_size),
                              chrom_params.mesh_x_list,
                              chrom_params.mesh_y_list) for _ in range(population_size)]

def IdxFind(arr, 
            val):
    return np.argmin(np.abs(arr - val))

def FitEval(chromosome : list, 
            chrom_params: ChromParams,
            field_params: StrayCalcParams,
            exp_params: ExpParams):
    """
    For an individual chromosome with the specificed `mesh_size`,
    1. Creates the MagnetMesh and the concomitant Meshed Micromagnet object
    2. Compute the stray field
    3. Compute the Quality factor Q
    """
    chrom = magnet.MeshedMicroMagnet(chromosome,
                                     **chrom_params.MmSimParams.simulation_parameter)
    try:
        chrom.SystemGen(chrom_params.z_gran, 
                        chrom_params.n_cpu)
        for comp in [0, 1, 2]:
            chrom.CalcStray(field_params.field_x_range, 
                            field_params.field_y_range, 
                            field_params.gran,
                            field_params.qunatum_dot_pos_z,
                            component = comp,
                            n_cpu = chrom_params.n_cpu)
        Field_trace = copy.deepcopy(chrom.field_trace)
        dx = (chrom.field_x[2] - chrom.field_x[1])
        dy = (chrom.field_y[2] - chrom.field_y[1])
        T2_star_trace = np.zeros((Field_trace.shape[0]-1, Field_trace.shape[1]-1))
        Q_trace = np.zeros_like(T2_star_trace)
        
        for i in range(Field_trace.shape[0]-1):
            for j in range(Field_trace.shape[1]-1):
                dB = np.sqrt(
                    ((Field_trace[i+1, j, 1] - Field_trace[i, j, 1]) / dx)**2 +
                    ((Field_trace[i, j+1, 1] - Field_trace[i, j, 1]) / dy)**2
                )
                # T2*:
                if dB > 1e-12:
                    T2_star = h*0.83255461115/np.pi/guB/dB/exp_params.sigma_pos
                else:
                    T2_star = 1e9  # some large number if gradient is extremely small
                
                # driving gradient for x and z components. Only the oscillation along the y axis is considered.
                dBg = np.sqrt(
                    ((Field_trace[i, j+1, 0] - Field_trace[i, j, 0]) / dy)**2 +
                    ((Field_trace[i, j+1, 2] - Field_trace[i, j, 2]) / dy)**2
                )
                Q_val = dBg * T2_star

                T2_star_trace[i, j] = T2_star
                Q_trace[i, j] = Q_val
        
        Q_total = 0
        for (qd_x, qd_y) in exp_params.qd_positions:
            idx_x = IdxFind(chrom.field_x, qd_x)
            idx_y = IdxFind(chrom.field_y, qd_y)
            Q_total += Q_trace[idx_x, idx_y]
        return Q_total
    
    except Exception as e:
        print("Simulation failed. Error: ", e)
        return -1e9
    
def TournamentSelection(population :list[magnet.MagnetMesh], #a population, list of a chromosome
                        fitnesses : list, #a list of a fitness, in the same order with the population
                        k = 3): #tournament size, default is 3
    subset_idx = np.random.choice(len(population), k, replace = False)
    best = None
    best_fit = -np.inf
    for i in subset_idx:
        if fitnesses[i] > best_fit:
            best_fit = fitnesses[i]
            best = population[i].mesh_coord
    return copy.deepcopy(best)

def UniformCrossOver(parent1,
                     parent2,
                     crossover_rate = 0.8): #default crossover rate is 0.8
    if random.random() > crossover_rate:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)
    for i in range(np.shape(parent1)[0]):
        for j in range(np.shape(parent1)[1]):
            if random.random() < 0.5:
                temp = child1[i,j]
                child1[i,j] = child2[i,j]
                child2[i,j] = temp
                
    return child1, child2

def RunGA(population_size : int,
          number_of_generations : float,
          chrom_params: ChromParams,
          field_params: StrayCalcParams,
          exp_params: ExpParams,
          crossover_rate : float = 0.8,
          mutation_rate : float = 0.001,
          mutation_constraint = None,
          init_is_random = True,
          init_input: magnet.MagnetMesh = None,
          input_mutation_rate = 0.001):
    # 1. Create initial population
    if init_is_random == True:
        population = InitChromGen(chrom_params,
                                  population_size)
    else:
        population = InitChromGen(chrom_params,
                                  population_size,
                                  init_is_random,
                                  init_input,
                                  input_mutation_rate
                                  )
        
    # Track best solution over all generations
    best_overall_chrom = None
    best_overall_fit = -np.inf

    # Track best solution of *each* generation
    best_per_generation = []
    
    for gen in range(number_of_generations):
        print(f"=== Generation {gen} ===")
        
        # 2. Evaluate fitness of all individuals
        fitnesses = []
        for _, indiv in enumerate(population):
            fit = FitEval(indiv,
                          chrom_params,
                          field_params,
                          exp_params)
            fitnesses.append(fit)
        
        # Find the best in this generation
        max_fit = max(fitnesses)
        max_idx = np.argmax(fitnesses)
        
        # Keep the best chromosome for this generation
        gen_best_chrom = copy.deepcopy(population[max_idx].mesh_coord)
        
        # Update global best if needed
        if max_fit > best_overall_fit:
            best_overall_fit = max_fit
            best_overall_chrom = copy.deepcopy(gen_best_chrom)
        
        print(f"Best fitness this generation = {max_fit:.6f}, Global best so far = {best_overall_fit:.6f}")
        
        # Store generation best info
        best_per_generation.append((gen, gen_best_chrom, max_fit))
        
        # 3. Create next generation
        new_population = []
        
        # (Optional) Elitism: preserve best from this generation automatically
        # new_population.append(gen_best_chrom)
        
        # Fill up new population
        while len(new_population) < population_size:
            # (a) selection
            p1 = TournamentSelection(population, fitnesses, k=3)
            p2 = TournamentSelection(population, fitnesses, k=3)
            
            # (b) crossover
            c1, c2 = UniformCrossOver(p1, p2, crossover_rate)
            
            # (c) mutation
            Mutate(c1, mutation_rate, mutation_constraint)
            Mutate(c2, mutation_rate, mutation_constraint)
            
            new_population.append(magnet.MagnetMesh(c1,
                                  chrom_params.mesh_x_list,
                                  chrom_params.mesh_y_list))
            
            if len(new_population) < population_size:
                new_population.append(magnet.MagnetMesh(c2,
                                                        chrom_params.mesh_x_list,
                                                        chrom_params.mesh_y_list))
        
        # Replace population
        population = new_population
    
    print("=== GA Finished ===")
    print(f"Best overall fitness = {best_overall_fit:.6f}")
    
    # Return the best individual found plus the history of best solutions
    return best_overall_chrom, best_overall_fit, best_per_generation
