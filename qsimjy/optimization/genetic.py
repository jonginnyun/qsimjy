import random
import numpy as np
import copy
import time
import datetime
import gc
import subprocess
from .. import magnet
from skimage import measure
from shapely.geometry import Polygon, Point

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
    
    Args:
        mesh_size (tuple): Size of the mesh (rows, colums).
    
    Returns: 
        np.ndarray: A binary (0 or 1) 2D array representing the mesh.
    """
    return np.random.randint(0, 2, size = mesh_size) 

    
def Mutate(chrom,
           mutation_rate = 0.001,#default mutation rate is 0.00: 1
           mut_constraint = None):
    """
    Mutates a chromosome (2D binary array). 
    
    Args:
        chrom (np.ndarray): The binary mesh to mutate.
        mutation_rate (float): Probability of flipping a cell.
        mut_constraint (str, optional): Mutation constraint. Options:
            - None: No constraint; mutate randomly
            - 'adj': Allow mutation only if adjacent cell supports it (has the same value to mutate).
            
    Returns:
        None. Mutates in place.
    """
    if mut_constraint == None:
        for i in range(np.shape(chrom)[0]):
            for j in range(np.shape(chrom)[1]):
                if random.random() < mutation_rate:
                    chrom[i,j] = 1-chrom[i,j]
                    
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
        chrom[:] = _mutated
        
    elif mut_constraint == 'boundary':
        _prev = copy.deepcopy(chrom)
        _mutated = copy.deepcopy(chrom)


def InitChromGen(chrom_params: ChromParams,
                 population_size : int,
                 is_random = True,
                 init_input: magnet.MagnetMesh = None,
                 mutation_rate = 0.001):
    """
    Generates an initial population of chromosomes.
    
    Args:
        chrom_params (ChromParams): Parameters for mesh and boundary.
        population_size (int): Number of chromosomes to generate for an initial population.
        is_random (bool): Whether to generate randomly (True) or use provided input (False).
        init_input (MagnetMesh, optional): Input mesh to mutate if not random.
        mutation_rate (float): Mutation rate if init_input is used.
    
    Returns:
        list: List of MagnetMesh objects.
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
    """
    Find the index in `arr` where the value is closest to `val`.
    
    Args:
        arr (np.ndarray): 1D array to search.
        val (float): Target value.
    
    Returns:
        int: Index of the closest value.
    """
    return np.argmin(np.abs(arr - val))

def FitEval(chromosome : list, 
            chrom_params: ChromParams,
            field_params: StrayCalcParams,
            exp_params: ExpParams):
    """
    For an individual chromosome with the specificed `mesh_size`, evaluates the fitness by:
        1. Creates the MagnetMesh and the concomitant Meshed Micromagnet object
        2. Compute the stray field using MuMax3 and a C++ backend.
        3. Compute the Quality factor Q over specific dot locations.
    
    Args:
        chromosome (np.ndarray): Binary mesh representation of a chromosome.
        chrom_params (ChromParams): Mesh/grid size and MuMax3 configuration.
        field_params (StrayCalcParams): Field calculation grid and depth.
        exp_params (ExpParams): Quantum dot positions and noise model
        
    Returns:
        float: The computed fitness (total Q value). Returns a large negative value if simulation fails
        
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
    '''
    Select the best chromosome among a random subset of the population using tournament selection.

    Args:
        population (list of MagnetMesh): Current population of individuals.
        fitnesses (list of float): Fitness scores corresponding to the population.
        k (int): Tournament size; number of individuals to sample.

    Returns:
        np.ndarray: Deep copy of the mesh of the best individual in the tournament.

    '''
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
    """
    Perform uniform crossover on two binary parent meshes to produce two offspring.

    Each gene (cell) in the offspring is swapped with a 50% chance, independently for each position.
    The crossover happens only if a random threshold is passed based on `crossover_rate`.

    Args:
        parent1 (np.ndarray): First parent binary mesh.
        parent2 (np.ndarray): Second parent binary mesh.
        crossover_rate (float): Probability that crossover occurs.

    Returns:
        tuple: Two offspring meshes (np.ndarray), each same shape as parents.
    """
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
    """
    Run a genetic algorithm to optimize binary micromagnet meshes.
    The polygon-based magnet shape uses the function defined below.

    This function initializes a population of chromosomes, iteratively evolves them
    using tournament selection, crossover, and mutation, and returns the best individual found.

    Args:
        population_size (int): Number of individuals in the population.
        number_of_generations (int): Number of generations to evolve.
        chrom_params (ChromParams): Parameters related to mesh size and simulation.
        field_params (StrayCalcParams): Parameters for stray field calculation.
        exp_params (ExpParams): Experimental configuration including quantum dot info.
        crossover_rate (float): Probability of performing crossover.
        mutation_rate (float): Mutation rate for offspring.
        mutation_constraint (str): Mutation rule ('adj', 'boundary', or None).
        init_is_random (bool): Whether to generate initial population randomly.
        init_input (MagnetMesh, optional): Optional template to initialize population.
        input_mutation_rate (float): Mutation rate applied to init_input if used.

    Returns:
        tuple: (best chromosome mesh, best fitness score, list of generation-best records)
    """
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

########################################################
####  Polygon-based GA algorithm #######################
########################################################
class MultiPolygonChrom:
    """
    A chromosome storing multiple polygons for the polygon-based genetic algorithm.
    Each polygon is represented as a list of (x, y) vertices.
    
    Attributes:
        polygons (list): List of polygons, where each polygon is a list of (x, y) vertices.
        mesh_size (tuple): Size of the rasterization grid (nx, ny).
        x_bound (tuple): Bounds in the x-direction in micrometers (x_min, x_max).
        y_bound (tuple): Bounds in the y-direction in micrometers (y_min, y_max).
    """
    def __init__(self, polygons, mesh_size, x_bound, y_bound):
        self.polygons = polygons
        self.mesh_size = mesh_size
        self.x_bound = x_bound
        self.y_bound = y_bound

    def copy(self):
        """
        Create a deep copy of the current chromosome.

        Returns:
            MultiPolygonChrom: A new object with duplicated data.
        """
        return MultiPolygonChrom(copy.deepcopy(self.polygons),
                                 self.mesh_size,
                                 self.x_bound,
                                 self.y_bound)

    def to_shapely_polygons(self):
        """
        Convert each polygon (list of vertices) into a Shapely Polygon.
        #--Vertices are sorted by angle around their centroid to reduce self-intersections.--currently commented out.
        
        Returns:
            list: A list of shapely.geometry.Polygon objects.
        """
        poly_list = []
        for pts in self.polygons:
            if len(pts) < 3:
                continue
            #cx = sum(p[0] for p in pts) / len(pts)
            #cy = sum(p[1] for p in pts) / len(pts)
            #pts_sorted = sorted(pts, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
            #poly_list.append(Polygon(pts_sorted))
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            poly_list.append(poly)
        return poly_list

    def to_mesh(self):
        """
        Rasterize the multi-polygon into a 2D binary array of shape (nx, ny).
        Returns:
            np.ndarray: A binary array of shape (nx, ny) with 1 if a cell is inside any polygon.
        """
        nx, ny = self.mesh_size
        x_min, x_max = self.x_bound
        y_min, y_max = self.y_bound
        poly_list = self.to_shapely_polygons()
        mesh = np.zeros((nx, ny), dtype=int)
        dx = (x_max - x_min) / nx
        dy = (y_max - y_min) / ny
        for i in range(nx):
            for j in range(ny):
                cx = x_min + (i + 0.5) * dx
                cy = y_min + (j + 0.5) * dy
                for poly in poly_list:
                    if poly.contains(Point(cx, cy)):
                        mesh[i, j] = 1
                        break
        return mesh

# --- Helper functions for initialization, mutation, and crossover ---
def mesh_to_polygons(mesh_coord, mesh_size, x_bound, y_bound, num_polygons):
    """
    Extract polygons from a binary mesh using the marching squares method (find_contours).
    The polygon vertices are based on cell boundaries (not cell centers).

    Args:
        mesh_coord (np.ndarray): 2D numpy array (shape = mesh_size) with binary values (0 or 1)
        mesh_size (tuple): tuple (nx, ny) corresponding to mesh_coord dimensions
        x_bound (tuple): tuple (x_min, x_max) in physical units
        y_bound (tuple): tuple (y_min, y_max) in physical units
        num_polygons (int): desired number of polygons to return

    Returns:
        list: A list of polygons, each represented as an ordered list of (x, y) vertices.
    """
    nx, ny = mesh_size
    # Use marching squares to extract contours (cell boundaries)
    contours = measure.find_contours(mesh_coord, level=0.5)

    # Calculate grid spacing based on cell boundaries.
    # Note: here we assume that the mesh grid covers [x_min, x_max] and [y_min, y_max] exactly,
    # so each cell has width dx and height dy.
    dx = (x_bound[1] - x_bound[0]) / nx
    dy = (y_bound[1] - y_bound[0]) / ny

    polygons = []
    for contour in contours:
        pts = []
        # Each contour point is given as (i, j) in array coordinates.
        # Since our mesh is defined such that index i corresponds to the x-direction and j to y,
        # we map: physical x = x_bound[0] + i * dx, physical y = y_bound[0] + j * dy.
        for point in contour:
            i, j = point
            x = x_bound[0] + i * dx
            y = y_bound[0] + j * dy
            pts.append((x, y))
        # Create a polygon from the contour and fix self-intersections if needed.
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        # If the result is a MultiPolygon, take the one with the largest area.
        if poly.geom_type == "MultiPolygon":
            best_poly = max(poly.geoms, key=lambda g: g.area)
            pts = list(best_poly.exterior.coords)
        else:
            pts = list(poly.exterior.coords)
        polygons.append(pts)

    # Sort polygons by area (largest first)
    def polygon_area(coords):
        return Polygon(coords).area

    polygons = sorted(polygons, key=polygon_area, reverse=True)
    return polygons[:num_polygons]

def resample_polygon_vertices(pts, num_vertices):
    """
    Resamples a polygon by the input number of vertices. locates a vertices such that the distortion is minimal.
    
    Args: 
        pts  (list): A list of polygon, with the format of [(x, y), ...]. The list should be open, i.e. first ≠ last.
        num_vertices (int): desired number of vertices (≥ 3)
    
    Returns:
        list: A list of points representing a polygon (a list of polygon).
    """
    # --- helper ---
    def edge_lengths(p):
        return [np.hypot(p[(i+1)%n][0]-p[i][0],
                         p[(i+1)%n][1]-p[i][1]) for i in range(n)]
    
    def triangle_area(pa, pb, pc):
        return abs( (pb[0]-pa[0])*(pc[1]-pa[1])
                  - (pb[1]-pa[1])*(pc[0]-pa[0]) ) * 0.5
    
    pts = list(pts) ; n = len(pts)
    if n < 3:
        raise ValueError("Polygon needs ≥3 vertices")
    if num_vertices == n:
        return pts.copy()

    while len(pts) > num_vertices:
        n = len(pts)
        costs = []
        for i in range(n):
            pa, pb, pc = pts[(i-1)%n], pts[i], pts[(i+1)%n]
            costs.append(triangle_area(pa, pb, pc))
        idx = int(np.argmin(costs))
        pts.pop(idx)
        
    while len(pts) < num_vertices:
        n = len(pts)
        lengths = edge_lengths(pts)
        idx = int(np.argmax(lengths))         
        pa, pb = pts[idx], pts[(idx+1)%n]
        mid = ((pa[0]+pb[0])*0.5, (pa[1]+pb[1])*0.5)
        pts.insert(idx+1, mid)

    return pts

def init_multipoly_population(pop_size, mesh_size, x_bound, y_bound,
                              num_polygons, num_vertices,
                              init_is_random=True, init_input=None):
    """
    Initialize a population of MultiPolygonChrom objects.
    
    If init_is_random is True, random polygons are generated.
    Otherwise, if init_input is provided (as a list of Shapely polygons), the exterior coordinates of each polygon 
    are extracted and then resampled to have exactly num_vertices vertices.
    
    Args:
        pop_size: The size of the population.
        mesh_size: Tuple (nx, ny) representing the grid size for rasterization.
        x_bound, y_bound: Tuples (x_min, x_max) and (y_min, y_max) representing the physical boundaries.
        num_polygons: The number of polygons to include in each chromosome.
        num_vertices: The desired number of vertices for each polygon.
        init_is_random: If True, generate random polygons; otherwise, use init_input.
        init_input: If provided and init_is_random is False, it is used as the base input.
                    It should be a list of Shapely Polygon objects.
    
    Returns:
        list: A list of MultiPolygonChrom objects representing the population.
    """
    population = []
    base_polygons = None

    if not init_is_random and (init_input is not None):
        if isinstance(init_input, list) and isinstance(init_input[0], Polygon):
            # If init_input is a list of Shapely polygons, extract the exterior coordinates.
            base_polygons = [list(poly.exterior.coords) for poly in init_input]
        else:
            # Otherwise, assume init_input has an attribute 'mesh_coord' and use a mesh-to-polygon conversion.
            base_polygons = mesh_to_polygons(init_input.mesh_coord,
                                             mesh_size, x_bound, y_bound,
                                             num_polygons)
        # Resample each polygon to have exactly num_vertices vertices.
        base_polygons = [resample_polygon_vertices(poly, num_vertices) for poly in base_polygons]

    for _ in range(pop_size):
        if init_is_random or (base_polygons is None):
            # Generate random polygons: each polygon consists of num_vertices random points.
            polygons = []
            for _p in range(num_polygons):
                pts = []
                for _v in range(num_vertices):
                    rx = random.uniform(x_bound[0], x_bound[1])
                    ry = random.uniform(y_bound[0], y_bound[1])
                    pts.append((rx, ry))
                polygons.append(pts)
        else:
            # Use a deep copy of base_polygons to ensure each chromosome is independent.
            polygons = copy.deepcopy(base_polygons)
        chrom = MultiPolygonChrom(polygons, mesh_size, x_bound, y_bound)
        population.append(chrom)
    return population


def mutate_multipoly_points(chrom, mutation_rate, mutation_scale, max_attempts=5):
    """
    Shift each vertex in each polygon with probability 'mutation_rate'
    by a random amount up to ±mutation_scale. After each mutation,
    check if the new polygon is topologically valid. If not, try up to max_attempts.
    If a valid mutation is not achieved, revert to the original vertex.
    
    Args:
        chrom: MultiPolygonChrom object whose vertices are to be mutated.
        mutation_rate: The probability of mutating each vertex.
        mutation_scale: Maximum absolute shift applied to each vertex.
        max_attempts: Maximum number of attempts per vertex to produce a valid polygon.
    
    Returns:
        None. The input chromosome is mutated in-place.
    """
    xmin, xmax = chrom.x_bound
    ymin, ymax = chrom.y_bound
    # Loop through each polygon in the chromosome.
    for p_idx, poly in enumerate(chrom.polygons):
        # Work on a copy of the vertex list for this polygon.
        original_poly = list(poly)
        for v_idx, (x, y) in enumerate(poly):
            if random.random() < mutation_rate:
                original = (x, y)
                valid_mutation = False
                attempts = 0
                while not valid_mutation and attempts < max_attempts:
                    # Apply a random mutation
                    dx = random.uniform(-mutation_scale, mutation_scale)
                    dy = random.uniform(-mutation_scale, mutation_scale)
                    x_new = max(min(x + dx, xmax), xmin)
                    y_new = max(min(y + dy, ymax), ymin)
                    # Create a new vertex list for testing
                    new_poly = list(chrom.polygons[p_idx])
                    new_poly[v_idx] = (x_new, y_new)
                    test_poly = Polygon(new_poly)
                    # If the mutated polygon is valid, accept the mutation.
                    if test_poly.is_valid:
                        valid_mutation = True
                        chrom.polygons[p_idx][v_idx] = (x_new, y_new)
                    else:
                        attempts += 1
                # If no valid mutation was found after max_attempts, revert the change.
                if not valid_mutation:
                    chrom.polygons[p_idx][v_idx] = original

def polygon_area_intersect(polyA, polyB):
    """
    Return the intersection area between two Shapely Polygons, with repair fallback.
    
    Attemps direct intersection; if GEOSException occurs, the function tries to repair polygon by 1. angle-sort 2. buffer(0)
    if repair fails, returns 0.0, which will be removed by selection procedure.
    
    Args:
        polyA (Polygon): First polygon.
        polyB (Polygon): Second polygon.

    Returns:
        float: Area of intersection.
    """
    def _repair(poly):
        if poly.is_valid:
            return poly

        # 1. angle sort and retrial
        try:
            pts = list(poly.exterior.coords[:-1])          
            sorted_pts = sort_vertices_by_angle(pts)
            fixed = Polygon(sorted_pts)
            if fixed.is_valid:
                return fixed
        except Exception:
            pass

        # 2. last resort; buffer(0)
        try:
            fixed = poly.buffer(0)
            if fixed.is_valid:
                return fixed
        except Exception:
            pass

        # if repair fails, returns the original polygon
        return poly

    # 1st trial of intersection
    try:
        inter = polyA.intersection(polyB)
        return 0.0 if inter.is_empty else inter.area
    except Exception as e:
        print(f"[Warn] intersection failed: {e}")
        pass

    # 2nd trial after the repairment.
    polyA_fixed = _repair(polyA)
    polyB_fixed = _repair(polyB)

    try:
        inter = polyA_fixed.intersection(polyB_fixed)
        return 0.0 if inter.is_empty else inter.area
    except Exception as e2:
        print(f"[Warn] intersection second-try failed: {e2}")
        return 0.0


def sort_vertices_by_angle(vertices):
    """
    Sort a list of (x, y) vertices by angle around their centroid.
    
    Args:
        vertices (list): List of (x, y) coordinates.

    Returns:
        list: Sorted list of vertices.
    """
    if len(vertices) < 3:
        return vertices
    cx = sum(v[0] for v in vertices) / len(vertices)
    cy = sum(v[1] for v in vertices) / len(vertices)
    return sorted(vertices, key=lambda p: np.arctan2(p[1]-cy, p[0]-cx))

def blend_polygon_vertices(ptsA, ptsB, alpha=0.5):
    """
    Blend two polygons' vertex lists via linear interpolation.
    Each vertex is blended with weight `alpha` from ptsA and (1-alpha) from ptsB.
    
    Args:
        ptsA (list): First list of (x, y) vertices.
        ptsB (list): Second list of (x, y) vertices.
        alpha (float): Blending weight.

    Returns:
        list: List of blended (x, y) vertices.
    """
    n = min(len(ptsA), len(ptsB))
    blended = []
    for i in range(n):
        x1, y1 = ptsA[i]
        x2, y2 = ptsB[i]
        bx = alpha * x1 + (1 - alpha) * x2
        by = alpha * y1 + (1 - alpha) * y2
        blended.append((bx, by))
    return blended

def crossover_multipoly_blending(parent1, parent2, crossover_rate=0.8, alpha=0.5):
    """
    Perform polygon-based crossover using geometric blending of overlapping regions.

    For each polygon in `parent1`, find the most overlapping polygon in `parent2`
    and generate blended offspring polygons. Unmatched polygons are copied directly.

    Args:
        parent1 (MultiPolygonChrom): First parent chromosome.
        parent2 (MultiPolygonChrom): Second parent chromosome.
        crossover_rate (float): Probability of performing crossover.
        alpha (float): Weighting factor for blending polygon vertices.

    Returns:
        tuple: Two child chromosomes (MultiPolygonChrom).
    """
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    child1 = parent1.copy()
    child2 = parent2.copy()

    polysA = parent1.polygons
    polysB = parent2.polygons
    nA = len(polysA)
    nB = len(polysB)

    shapelyA = [Polygon((pts)) for pts in polysA]
    shapelyB = [Polygon((pts)) for pts in polysB]

    usedB = set()
    new_child1_polys = copy.deepcopy(polysA)
    new_child2_polys = copy.deepcopy(polysB)

    for iA in range(nA):
        max_area = 0.0
        best_jB = None
        for jB in range(nB):
            if jB in usedB:
                continue
            area_ij = polygon_area_intersect(shapelyA[iA], shapelyB[jB])
            if area_ij > max_area:
                max_area = area_ij
                best_jB = jB
        if (best_jB is not None) and (max_area > 1e-12):
            usedB.add(best_jB)
            va = (polysA[iA])
            vb = (polysB[best_jB])
            blendA = blend_polygon_vertices(va, vb, alpha)
            blendB = blend_polygon_vertices(vb, va, alpha)
            new_child1_polys[iA] = blendA
            new_child2_polys[best_jB] = blendB

    child1.polygons = new_child1_polys
    child2.polygons = new_child2_polys
    return child1, child2

def fit_eval_multipoly(chrom,            # <-- renamed for clarity
                       chrom_params,
                       field_params,
                       exp_params,
                       penalty_weight = 1,
                       min_thresh_T=0.050, 
                       margin=0.020):
    """
    Evaluate the fitness of a MultiPolygonChrom chromosome using local field gradients.

    For each quantum dot position, calculates local stray field gradients and
    derives quality factor Q. Penalizes inconsistent magnetic field magnitudes across QDs.
    
    To be specific, the penalty is introduced to allow the minimal addressibility, which is set by min_thresh_T and margin.
    The unit of both are Tesla.

    Args:
        chrom (MultiPolygonChrom): Chromosome encoding polygon geometry.
        chrom_params (ChromParams): Mesh and MuMax3 parameters.
        field_params (StrayCalcParams): Field range and granularity.
        exp_params (ExpParams): QD positions and noise parameters.
        penalty_weight (float): Weight of the penalty term.
        min_thresh_T (float): Minimum field norm to trigger penalty (Tesla).
        margin (float): Tolerance margin in Tesla.

    Returns:
        float: Fitness score (Q value with penalties).
    """
    # — 1. build the micromagnet once —
    mesh_arr = chrom.to_mesh()
    mmesh    = magnet.MagnetMesh(mesh_arr,
                                 chrom_params.mesh_x_list,
                                 chrom_params.mesh_y_list)
    mm = magnet.MeshedMicroMagnet(mmesh,
                                  **chrom_params.MmSimParams.simulation_parameter)

    try:
        mm.SystemGen(chrom_params.z_gran,
                     chrom_params.n_cpu)           

        _time = time.time()
        # --- geometry of the local 3×3 stencil -----------------------------
        # keep the same spatial step that field_params.gran implied
        full_dx = (field_params.field_x_range[1] - field_params.field_x_range[0]) \
                  / (field_params.gran[0] - 1)
        full_dy = (field_params.field_y_range[1] - field_params.field_y_range[0]) \
                  / (field_params.gran[1] - 1)

        h_const  = 4.13e-15
        guB_const = 2 * 57.8e-6
        Q_total = 0.0
        Bvec_list = []
        external_B = 4 * np.pi * 1e-7 * np.array(mm.Hext) # Tesla
        # — 2. loop over each requested quantum-dot position —
        for (qd_x, qd_y) in exp_params.qd_positions:

            # build a local 3-point grid in each direction (left, centre, right)
            local_x = (qd_x - full_dx, qd_x + full_dx)
            local_y = (qd_y - full_dy, qd_y + full_dy)
            local_gran = (3, 3)                      # 3 points per axis

            # allocate a fresh trace buffer for this dot
            mm.field_trace = np.zeros((*local_gran, 3))

            for comp in (0, 1, 2):                   # Bx, By, Bz
                mm.CalcStray(local_x,
                             local_y,
                             local_gran,
                             field_params.qunatum_dot_pos_z,
                             component=comp,
                             n_cpu=chrom_params.n_cpu)

            # shorthand for readability
            Fx = mm.field_trace[:, :, 0]
            Fy = mm.field_trace[:, :, 1]
            Fz = mm.field_trace[:, :, 2]
            Bvec_list.append(mm.field_trace[1,1,:]+external_B)
            # gradients around the central point (index 1,1)
            dBy_dx = (Fy[2, 1] - Fy[0, 1]) / (2 * full_dx)
            dBy_dy = (Fy[1, 2] - Fy[1, 0]) / (2 * full_dy)
            dB  = np.hypot(dBy_dx, dBy_dy)           # magnitude ‖∇By‖

            # driving-field gradient in x & z while oscillating along y
            dBdrive_dx = (Fx[2, 1] - Fx[0, 1]) / (2 * full_dx)
            dBdrive_dz = (Fz[1, 2] - Fz[1, 0]) / (2 * full_dy)
            dBg = np.hypot(dBdrive_dx, dBdrive_dz)

            T2_star = (h_const * 0.83255461115 /
                       (np.pi * guB_const * dB * exp_params.sigma_pos)
                       if dB > 1e-12 else 1e9)



            Q_total += dBg * T2_star
            mu_0 = 4 * np.pi * 1e-7  # N/A^2
        #Penalty Calculation. Can be commented out as a whole--------------------------------------------
        def norm_penalty(Bvec_list: list[np.ndarray],
                        min_thresh_T: float = 0.05,
                        margin: float = 0.003) -> float:
            norms = [np.linalg.norm(Bvec) for Bvec in Bvec_list]
            penalty = 1.0
            for i in range(len(norms)):
                for j in range(i + 1, len(norms)):
                    delta = abs(norms[i] - norms[j])
                    if delta < min_thresh_T:
                        penalty *= np.exp(-(delta / margin) ** 2)
            return 1-penalty
        # soft penalty application
        penalty_val = norm_penalty(Bvec_list, min_thresh_T=min_thresh_T, margin=margin)
        fitness = Q_total * (1 - penalty_weight * penalty_val)

        #------------------------------------------------------------------------------------------------
        print('Total time elapsed for a fitness evaluation:', datetime.timedelta(seconds=time.time() - _time))
        return fitness

    except Exception as err:
        print("Simulation failed in fit_eval_multipoly_fast:", err)
        return -1e9

def fit_eval_multipoly_robust(chrom: MultiPolygonChrom,
                               chrom_params: ChromParams,
                               field_params: StrayCalcParams,
                               exp_params: ExpParams,
                               num_samples: int = 3,
                               blur_sigma: float = 0.01,
                               penalty_weight = 1,
                               min_thresh_T=0.050, 
                               margin=0.020):
    """
    Robust fitness evaluation with fabrication-induced perturbation.

    Applies Gaussian offsets (buffering) to each polygon and evaluates
    average fitness across perturbed samples to simulate fabrication variation.

    Args:
        chrom (MultiPolygonChrom): Chromosome to evaluate.
        chrom_params (ChromParams): Parameters for simulation.
        field_params (StrayCalcParams): Grid and depth settings.
        exp_params (ExpParams): Quantum dot configuration.
        num_samples (int): Number of perturbed samples to evaluate.
        blur_sigma (float): Standard deviation of the shape deformation (µm).
        penalty_weight (float): Field uniformity penalty weight.
        min_thresh_T (float): Threshold for field strength.
        margin (float): Margin for penalty decay.

    Returns:
        float: Average fitness over perturbations.
    """
    _time = time.time()
    total_fitness = 0.0

    for _ in range(num_samples):
        try:
            # Sample offset from Gaussian: realistically models CD variation
            offset = np.random.normal(loc=0.0, scale=blur_sigma)
            
            # Apply shapely buffer
            deformed_polys = []
            for poly in chrom.to_shapely_polygons():
                # Buffer may return empty or invalid geometry
                try:
                    buffered = poly.buffer(offset)
                    if buffered.is_empty:
                        continue
                    # If MultiPolygon returned, split and keep biggest
                    if buffered.geom_type == "MultiPolygon":
                        buffered = max(buffered.geoms, key=lambda g: g.area)
                    if buffered.is_valid and isinstance(buffered, Polygon):
                        deformed_polys.append(buffered)
                except Exception as e:
                    print("[warn] buffer failed:", e)
                    continue
            
            if not deformed_polys:
                return -1e9  # Skip totally degenerate samples
            
            # Convert to MultiPolygonChrom
            def_poly_coords = [list(p.exterior.coords) for p in deformed_polys]
            deformed_chrom = MultiPolygonChrom(def_poly_coords,
                                               chrom.mesh_size,
                                               chrom.x_bound,
                                               chrom.y_bound)

            # Evaluate fitness of this perturbed version
            fit = fit_eval_multipoly(deformed_chrom,
                                     chrom_params,
                                     field_params,
                                     exp_params,
                                     penalty_weight,
                                     min_thresh_T, 
                                     margin)
            total_fitness += fit

        except Exception as e:
            print("Buffer or fitness failed:", e)
            total_fitness += -1e9
    del deformed_polys, deformed_chrom
    print('Individual chromosome fitness evaluation finished:', datetime.timedelta(seconds=time.time() - _time))
    return total_fitness / num_samples

# Tournament Selection
def tournament_selection_multipoly(population, fitnesses, k=3):
    """
    Tournament selection for MultiPolygonChrom. Returns a copy of the best chromosome among k random picks.
    """
    idxs = np.random.choice(len(population), k, replace=False)
    best_idx = None
    best_fit = -np.inf
    for i in idxs:
        if fitnesses[i] > best_fit:
            best_fit = fitnesses[i]
            best_idx = i
    return population[best_idx].copy()

#################################################
### MAIN GA FUNCTION#############################
#################################################
def RunGA_MultiPolygonBlendingPoints(population_size,
                                     number_of_generations,
                                     chrom_params,
                                     field_params,
                                     exp_params,
                                     consider_fab_blur = False, #takes fab blur into account; sigma is 10 nm (default)
                                     num_samples: int = 3,
                                     blur_sigma: float = 0.01,
                                     num_polygons=1,
                                     num_vertices=8,
                                     init_is_random=True,
                                     init_input=None,
                                     crossover_rate=0.8,
                                     alpha=0.5,
                                     mutation_rate=0.005,
                                     mutation_scale=0.01,
                                     k_tournament=3,
                                     penalty_weight = 1,
                                     min_thresh_T=0.050, 
                                     margin=0.020):
    """
    Genetic Algorithm for polygon-based chromosome evolution.

    Uses crossover via geometric blending of overlapping polygons,
    mutation via vertex jittering, and optional robustness to fabrication.

    Args:
        population_size (int): Size of the chromosome pool.
        number_of_generations (int): Number of GA iterations.
        chrom_params (ChromParams): Simulation settings.
        field_params (StrayCalcParams): Field grid and range.
        exp_params (ExpParams): Experimental QD configuration.
        consider_fab_blur (bool): Whether to simulate CD variation.
        num_samples (int): Perturbed samples per chromosome.
        blur_sigma (float): Fabrication blur standard deviation.
        num_polygons (int): Number of polygons per individual.
        num_vertices (int): Number of vertices per polygon.
        init_is_random (bool): Whether to generate population from scratch.
        init_input: Template MagnetMesh or polygon list.
        crossover_rate (float): Probability of crossover.
        alpha (float): Blending strength in crossover.
        mutation_rate (float): Per-point mutation probability.
        mutation_scale (float): Max vertex perturbation (µm).
        k_tournament (int): Tournament size.
        penalty_weight (float): Field uniformity penalty weight.
        min_thresh_T (float): Penalty activation threshold.
        margin (float): Soft penalty decay.

    Returns:
        tuple: (best chromosome, best fitness, list of (gen, best chrom, fit)).
    """
    _timeGA = time.time()
    mesh_size = chrom_params.mesh_size
    x_bound = chrom_params.mesh_x_list
    y_bound = chrom_params.mesh_y_list
    population = init_multipoly_population(population_size,
                                           mesh_size,
                                           x_bound,
                                           y_bound,
                                           num_polygons,
                                           num_vertices,
                                           init_is_random,
                                           init_input)
    best_overall_chrom = None
    best_overall_fit = -np.inf
    best_history = []

    for gen in range(number_of_generations):
        _timeGAloop = time.time()
        print(f"=== MultiPolygonBlending GA Generation {gen} ===")
        fitnesses = []
        for indiv in population:
            if consider_fab_blur == True:
                f_val = fit_eval_multipoly_robust(indiv, chrom_params, field_params, exp_params, num_samples = num_samples, blur_sigma = blur_sigma, penalty_weight = penalty_weight,min_thresh_T=min_thresh_T, margin=margin)
            else:
                f_val = fit_eval_multipoly(indiv, chrom_params, field_params, exp_params)
            fitnesses.append(f_val)
        gen_best_fit = max(fitnesses)
        gen_best_idx = np.argmax(fitnesses)
        if gen_best_fit > best_overall_fit:
            best_overall_fit = gen_best_fit
            best_overall_chrom = population[gen_best_idx].copy()
        print(f"Gen best = {gen_best_fit:.6f}, Global best = {best_overall_fit:.6f}")
        
        print('GA ONE EPOCH FINISHED:', datetime.timedelta(seconds=time.time() - _timeGAloop))
        best_history.append((gen, best_overall_chrom.copy(), gen_best_fit))
        new_population = []
        while len(new_population) < population_size:
            p1 = tournament_selection_multipoly(population, fitnesses, k_tournament)
            p2 = tournament_selection_multipoly(population, fitnesses, k_tournament)
            c1, c2 = crossover_multipoly_blending(p1, p2, crossover_rate, alpha)
            mutate_multipoly_points(c1, mutation_rate, mutation_scale)
            mutate_multipoly_points(c2, mutation_rate, mutation_scale)
            new_population.append(c1)
            if len(new_population) < population_size:
                new_population.append(c2)
        population = new_population
        del new_population
        gc.collect()
        #subprocess.run(["killall","-q","mumax3"])

    print("=== MultiPolygonBlending GA Finished ===")
    print(f"Best overall fitness = {best_overall_fit:.6f}")
    print('GA FINIALLY FISINISHED WITH THE FOLLOWING CALCULATION TIME:', datetime.timedelta(seconds=time.time() - _timeGA))
    return best_overall_chrom, best_overall_fit, best_history