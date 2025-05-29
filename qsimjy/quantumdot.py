from __future__ import annotations

import os
import time
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import shapely as shp
import ezdxf
import _cxx_potcalc
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from shapely.plotting import plot_polygon, plot_points
from shapely.strtree import STRtree
from shapely.ops import unary_union
from multiprocess import Pool

"""
CAD gate parsing and electrostatic simulation utilities.
Supports DXF file format and maps gate layers to physical potential contributions.
"""
def merge_overlapping_gate_patterns(gate_set: Gate_set, 
                                    name_list: List[string] = None): 
    """
    Merge Gate_pattern which overlaps. Returns a new list of Gate_pattern objects where:
    1. Overlapping gate patterns are unioned.
    2. For each overlapping gate group, the first gate's name is used as the merged Gate_pattern object.
    
    Args:
        gate_set (Gate_set): Gate_set object containing Gate_patterns.
        name_list (List[string]): A list of gate_set names to be merged.
        
    Returns:
        merged_gate_set (Gate_set]): Gate_set object of Gate_patterns after the merging of overlapping gates is done.
    """
    if name_list == None:
        Exception("input name_list")
    gate_patterns = gate_set.gate_list
    polys = [gp.gate_polygon for gp in gate_patterns if gp.name in name_list]
    gps = [gp for gp in gate_patterns if gp.name in name_list]
    others = [gp for gp in gate_patterns if gp.name not in name_list]
    tree = STRtree(polys) #founds polygons that could possibly overlap with a given gate_polygon
    used = [False] * len(polys)
    merged_patterns: List[Gate_pattern] = []
    
    for i, poly in enumerate(polys):
        if used[i]:
            continue
        candidates = [j for j in tree.query(poly) if not used[j]]
        group_idxs = [j for j in candidates if poly.intersects(polys[j])]
        for j in group_idxs:
            used[j] = True
        unioned  = unary_union([poly]+[polys[j] for j in group_idxs]).buffer(0)
        merged_gate = Gate_pattern()
        merged_gate.set_polygon(unioned)
        merged_gate.set_name(gps[i].name)
        merged_patterns.append(merged_gate)
    # for i, gate in enumerate(gate_patterns):
    #     if used[i]:
    #         continue
    #     merged_patterns.append(gate)
    merged_patterns += others
    merged_gate_set = Gate_set()
    merged_gate_set.add_gates(merged_patterns)
    return merged_gate_set
        

    
def dxf_exporter(file_dir: str, 
                 offset=0.01,
                 cap_style: {'round','mitre','bevel'}  = 'round',
                 join_style: {'round','mitre','bevel'} = 'round'):
    """
    Import a DXF file and convert each LWPOLYLINE entity into a Gate_pattern object.
    Each Gate_pattern is added to a Gate_set.

    Args:
        file_dir (str): Path to the DXF file.
        offset (float): Shape offset due to the finite resolution of the lithography equipment.

    Returns:
        Gate_set: A set of Gate_pattern objects created from the DXF file.
    """
    doc = ezdxf.readfile(file_dir)
    polygon_list = []
    for entity in doc.modelspace().query('LWPOLYLINE'):
        vertices = [(vertex[0], vertex[1]) for vertex in entity.vertices()]
        if len(vertices) >= 4:
            g = Gate_pattern()
            g.add_points(vertices, offset, cap_style, join_style)
            g.set_name(entity.dxf.layer)
            polygon_list.append(g)
    gate_set = Gate_set()
    gate_set.add_gates(polygon_list)
    return gate_set


def gate_level_setter(gate_set, level_dict: dict) -> None:
    """
    Set the gate levels for each gate in the gate_set based on the level_dict.

    Args:
        gate_set (Gate_set): The set of gates to update.
        level_dict (dict): A dictionary mapping gate names to levels.
    """
    for gate in gate_set.gate_list:
        if gate.name in level_dict:
            gate.gate_level = level_dict[gate.name]


def Green_Function_der(x, y, z, xp, yp, zp) -> float:
    """
    Calculate the derivative of the 2D Green's Function for a quantum dot well geometry.

    Args:
        x (float): x-coordinate of target.
        y (float): y-coordinate of target.
        z (float): z-coordinate of target.
        xp (float): x-coordinate of source.
        yp (float): y-coordinate of source.
        zp (float): z-coordinate of source.

    Returns:
        float: The value of the derivative of the Green's function.
    """
    dx = x - xp
    dy = y - yp
    r_plus = np.sqrt(dx**2 + dy**2 + (z + zp)**2)
    r_minus = np.sqrt(dx**2 + dy**2 + (z - zp)**2)
    z_plus = zp + z
    z_minus = zp - z
    return -z_minus / r_minus**3 + z_plus / r_plus**3


class Gate_pattern:
    """
    Gate pattern Class.

    Stores the information on the gate geometry as a set of points.
    The order of points to be added to the gate pattern object should be either clockwise
    or counter-clockwise along the demarcation of the gate geometry.

    Attributes:
        point_list_x (list): x-coordinates of the gate polygon vertices.
        point_list_y (list): y-coordinates of the gate polygon vertices.
        gate_level (int): Stacking sequence of the gate; lower value lies beneath higher ones.
        voltage (float): Voltage applied to the gate.
        gate_polygon (Polygon): Shapely polygon object representing the gate.
        name (str): Name of the gate.
    """

    _number_of_patterns = 0

    def __init__(self):
        Gate_pattern._number_of_patterns += 1
        self.point_list_x = []
        self.point_list_y = []
        self.gate_level = 0
        self.voltage = 0
        self.gate_polygon = None
        self.name = "default"

    def add_point(self, point_arrays):
        """
        Add points to the gate pattern.

        Args:
            point_arrays (list): List of (x, y) lists (or arrays) representing the vertices.
        """
        if isinstance(point_arrays, (np.ndarray, list)):
            self.point_list_x.append(point_arrays[0])
            self.point_list_y.append(point_arrays[1])
            polygon_points = list(zip(self.point_list_x, self.point_list_y))
            self.gate_polygon = shp.Polygon(polygon_points)
        else:
            raise TypeError(
                "Points to be added should be a list or np.array with the form of [x, y]"
            )

    def add_points(self, 
                   point_tuple_list, 
                   offset=0, 
                   cap_style = None,
                   join_style = None) -> None:
        """
        Add multiple points to the gate pattern.

        Args:
            point_tuple_list (list): List of (x, y) tuples representing the vertices.
            offset (float): Offset to apply to the polygon shape.
        """
        if not isinstance(point_tuple_list, (list, np.ndarray)):
            raise TypeError(
                "Points to be added should be a list or np.array of (x, y) tuples"
            )
        for point in point_tuple_list:
            if not (isinstance(point, tuple) and len(point) == 2):
                raise ValueError("Every element should be a tuple in the form of (x, y)")
            self.point_list_x.append(point[0])
            self.point_list_y.append(point[1])
        polygon_points = list(zip(self.point_list_x, self.point_list_y))
        self.gate_polygon = shp.Polygon(polygon_points)
        if offset != 0:
            self.gate_polygon = self.gate_polygon.buffer(offset, 
                                                         cap_style = cap_style,
                                                         join_style = join_style)

    def set_polygon(self, polygon: Polygon):
        """
        Set the gate polygon directly from a shapely Polygon object.

        Args:
            polygon (Polygon): The shapely Polygon object to set.
        """
        self.point_list_x = list(polygon.exterior.coords.xy[0])
        self.point_list_y = list(polygon.exterior.coords.xy[1])
        self.gate_polygon = polygon

    def plot(self, color=None, ax=None) -> None:
        """
        Plot the gate polygon.

        Args:
            color: Color of the polygon.
            ax: Matplotlib Axes object to plot on.
        """
        plot_polygon(self.gate_polygon, color=color, add_points=False, ax=ax)

    def set_level(self, gate_level) -> None:
        self.gate_level = gate_level

    def set_voltage(self, voltage) -> None:
        self.voltage = voltage

    def set_name(self, name: str) -> None:
        self.name = name

    def __del__(self) -> None:
        Gate_pattern._number_of_patterns -= 1


class Gate_set:
    """
    Gate set Class.

    Stores a list of Gate_pattern objects.
    """

    def __init__(self):
        self.gate_list = []

    def add_gate(self, gate_object):
        if not isinstance(gate_object, Gate_pattern):
            raise TypeError("The gate to be added should be a Gate_pattern object")
        self.gate_list.append(gate_object)

    def remove_gate(self, gate_index):
        if not isinstance(gate_index, int):
            raise TypeError("The gate index should be an integer")
        del self.gate_list[gate_index]

    def add_gates(self, gate_object_list):
        if not isinstance(gate_object_list, list):
            raise TypeError("Gates to be added should be a list of Gate_pattern objects")
        for gate in gate_object_list:
            if not isinstance(gate, Gate_pattern):
                raise TypeError(
                    "Every element of the gate list should be a Gate_pattern object"
                )
            self.gate_list.append(gate)

    def plot(self):
        cmap = matplotlib.colormaps["Spectral"]
        num_gates = len(self.gate_list)
        for i, gate in enumerate(self.gate_list):
            gate.plot(color=cmap(i / num_gates))


class Quantum_dot_device:
    """
    Quantum dot device Class.

    Stores gate objects and information on the gate thickness, quantum well thickness,
    and quantum well location.
    """

    def __init__(self):
        self.gate_set = Gate_set()
        self.quantum_well_depth = 0
        self.quantum_well_width = 0
        self.boundary_points = []
        self.potential_value = None
        self.potential_xlist = None
        self.potential_ylist = None
        self.initial_oxide_thickness = None
        self.oxide_thickness = None

    def add_gate_set(self, gate_set):
        if not isinstance(gate_set, Gate_set):
            raise TypeError("Gate set to be added should be a Gate_set object")
        self.gate_set = gate_set

    def set_well_width(self, width):
        self.quantum_well_width = width

    def set_well_depth(self, depth):
        self.quantum_well_depth = depth

    def set_boundary(self, boundary_points):
        """
        Set the boundary of the quantum dot device.

        Parameters:
        - boundary_points (list): List of four (x, y) tuples that demarcate the boundary.
        """
        if not (isinstance(boundary_points, list) and len(boundary_points) == 4):
            raise ValueError("Boundary points should be a list of four (x, y) tuples")
        for point in boundary_points:
            if not (isinstance(point, tuple) and len(point) == 2):
                raise ValueError("Every element should be a tuple in the form of (x, y)")
        self.boundary_points = boundary_points

    def plot(self):
        """
        Plot the quantum dot device, including the gates and boundary.
        """
        if not self.boundary_points:
            raise ValueError("Boundary should be set before plotting")
        fig, ax = plt.subplots()
        cmap = matplotlib.colormaps["Spectral"]
        x, y = zip(*self.boundary_points)
        ax.set_ylim([np.min(y), np.max(y)])
        ax.set_xlim([np.min(x), np.max(x)])
        boundary_polygon = shp.Polygon(self.boundary_points)
        num_gates = len(self.gate_set.gate_list)
        for i, gate in enumerate(self.gate_set.gate_list):
            gate.plot(color=cmap(i / (num_gates + 1)), ax=ax)
        plot_polygon(
            boundary_polygon,
            color=cmap(num_gates / (num_gates + 1)),
            add_points=False,
            ax=ax,
        )

    def plot_by_level(self):
        """
        Plot the quantum dot device gates colored by their gate level.
        """
        if not self.boundary_points:
            raise ValueError("Boundary should be set before plotting")
        fig, ax = plt.subplots()
        cmap = matplotlib.colormaps["Spectral"]
        x, y = zip(*self.boundary_points)
        ax.set_ylim([np.min(y), np.max(y)])
        ax.set_xlim([np.min(x), np.max(x)])
        levels = [gate.gate_level for gate in self.gate_set.gate_list]
        spec = max(levels) - min(levels) if levels else 1
        for gate in self.gate_set.gate_list:
            gate.plot(color=cmap(gate.gate_level / spec), ax=ax)

    def boundary_voltage(self, x, y):
        voltage_list = []
        level_list = []
        for gate in self.gate_set.gate_list:
            if gate.gate_polygon.contains(Point(x, y)):
                voltage_list.append(gate.voltage)
                level_list.append(gate.gate_level)
        if level_list:
            min_index = level_list.index(min(level_list))
            return voltage_list[min_index]
        return 0

    def gate_level(self, x, y):
        levels = [
            gate.gate_level
            for gate in self.gate_set.gate_list
            if gate.gate_polygon.contains(Point(x, y))
        ]
        return min(levels) if levels else float("inf")

    def calc_potential(
        self,
        x_list_plot,
        y_list_plot,
        x_list_sum,
        y_list_sum,
        number_of_meshes,
        gran_list,
        n_cpu=-1,
    ):
        """
        Calculate the potential over a grid using multiprocessing.

        Parameters:
        - x_list_plot (list): [xmin, xmax] for plotting.
        - y_list_plot (list): [ymin, ymax] for plotting.
        - x_list_sum (list): [xmin, xmax] for summation.
        - y_list_sum (list): [ymin, ymax] for summation.
        - number_of_meshes (int): Number of meshes for the lookup table.
        - gran_list (list): [gran_x, gran_y], grid sizes for plotting.
        - n_cpu (int): Number of CPUs to use. Default is all available CPUs.

        Note:
        This method assumes that child processes can access global variables, as in Linux.
        The code may not run in Windows.
        """
        if (
            len(x_list_sum) != 2
            or len(y_list_sum) != 2
            or len(x_list_plot) != 2
            or len(y_list_plot) != 2
        ):
            raise ValueError("x and y lists should be length-2 lists")
        if len(gran_list) != 2:
            raise ValueError(
                "gran_list must be a length-2 list, in the order of gran_x and gran_y"
            )
        if self.initial_oxide_thickness is None:
            raise ValueError("Input initial_oxide_thickness")
        if self.oxide_thickness is None:
            raise ValueError("Input oxide_thickness")

        x_list = np.linspace(x_list_sum[0], x_list_sum[1], number_of_meshes)
        y_list = np.linspace(y_list_sum[0], y_list_sum[1], number_of_meshes)
        _delta_xy = (x_list[2] - x_list[1]) * (y_list[2] - y_list[1])
        gate_potential_trace_lookup = np.zeros((number_of_meshes, number_of_meshes))
        gate_level_trace_lookup = np.zeros((number_of_meshes, number_of_meshes))

        if n_cpu == -1:
            n_cpu = os.cpu_count()
        elif n_cpu >= os.cpu_count():
            raise ValueError(
                "The number of CPUs (n_cpu) for multiprocessing exceeds the available cores"
            )

        # Helper function for the first multiprocessing step
        def _multiproc_1(args):
            k, j = args
            x = x_list[j]
            y = y_list[k]
            potential = self.boundary_voltage(x, y)
            level = self.gate_level(x, y)
            return k, j, potential, level
        
        if __name__ != '__mp_main__':
            # Generate lookup table
            start = time.time()
            _args_list = [
                (k, j) for k in range(number_of_meshes) for j in range(number_of_meshes)
            ]
            print("Lookup table generation initiated")
            with Pool(n_cpu) as pool:
                results = pool.map(_multiproc_1, _args_list)
            for k, j, potential, level in results:
                gate_potential_trace_lookup[j, k] = potential
                gate_level_trace_lookup[j, k] = level
            print("Lookup table generation completed")

            # Prepare for potential calculation
            _gran_x, _gran_y = gran_list
            self.potential_xlist = np.linspace(x_list_plot[0], x_list_plot[1], _gran_x)
            self.potential_ylist = np.linspace(y_list_plot[0], y_list_plot[1], _gran_y)
            # Contiguous array generation for a C++ implementation
            x_list = np.ascontiguousarray(x_list, dtype=np.float64)
            y_list = np.ascontiguousarray(y_list, dtype=np.float64)
            x_plist = np.ascontiguousarray(self.potential_xlist, dtype=np.float64)
            y_plist = np.ascontiguousarray(self.potential_ylist, dtype=np.float64)
            gate_potential_trace_lookup = np.ascontiguousarray(gate_potential_trace_lookup, dtype=np.float64)
            gate_level_trace_lookup = np.ascontiguousarray(gate_level_trace_lookup, dtype=np.float64)
            answer_trace = np.ascontiguousarray(np.zeros((_gran_x, _gran_y)), dtype=np.float64)
            _cxx_potcalc.potcalc(answer_trace, gate_potential_trace_lookup, gate_level_trace_lookup, x_plist,y_plist, x_list, y_list, [self.initial_oxide_thickness, self.oxide_thickness, self.quantum_well_depth], n_cpu)
            end = time.time()
            print("Finished. Total time elapsed:", datetime.timedelta(seconds=end - start))

        self.potential_value = answer_trace
