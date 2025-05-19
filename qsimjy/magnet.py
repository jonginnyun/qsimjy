from __future__ import annotations

import os
import subprocess
import tempfile
import time
import datetime
from types import SimpleNamespace
from typing import List, Tuple
from pathlib import Path

import numpy as np
import discretisedfield as df  # for OVF reading / mesh handling
from shapely.geometry import Point

import _cxx_magcalc  # C++ stray‑field kernel
'''
_mumax_server_address = "127.0.0.1:8080"
_mumax_server = subprocess.Popen(
    ["mumax3", "-server", _mumax_server_address],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
time.sleep(2)

atexit.register(lambda: (_mumax_server.terminate(), _mumax_server.wait()))
'''

# Scientific Constants
conversionfactor_of_mT_into_Am = 0.7957747154594767e3  # mT → A/m, ref;https://maurermagnetic.com/en/demagnetizing/technology/convert-magnetic-units/
mu_0 = 4 * np.pi * 1e-7  # vacuum permeability [N * A^-2]
prop_const = mu_0 / np.pi / 4 # proportionality constant for magnetic field

def _make_mask(mesh: df.Mesh, gates, x_min, y_min):
    """
    Generate a 3D boolean mask indicating regions within gate polygons.

    Args:
        mesh (df.Mesh): DiscretisedField mesh.
        gates (list): List of gate objects with `gate_polygon` attribute.
        x_min (float): Minimum x coordinate (µm).
        y_min (float): Minimum y coordinate (µm).

    Returns:
        np.ndarray: Boolean array (nx, ny, nz) marking gate regions as True.
    """
    nx, ny, nz = mesh.n
    mask = np.zeros((nx, ny, nz), dtype=bool)
    for gate in gates:
        for ix, x in enumerate(mesh.cells.x):
            for iy, y in enumerate(mesh.cells.y):
                if gate.gate_polygon.contains(Point(x*1e6 , y*1e6)):
                    mask[ix, iy, :] = True
    return mask

def _make_mask_mesh(mesh: df.Mesh, meshed_micromagnet, x_min, y_min):
    """
    Generate 3D boolean mask from a binary mesh_coord.

    Args:
        mesh (df.Mesh): DiscretisedField mesh.
        meshed_micromagnet (MagnetMesh): Object containing `mesh_coord`.
        x_min (float): Minimum x (µm).
        y_min (float): Minimum y (µm).

    Returns:
        np.ndarray: Boolean array (nx, ny, nz) with True where mesh_coord != 0.
    """
    nx, ny, nz = mesh.n
    mask = np.zeros((nx, ny, nz), dtype=bool)
    for ix, x in enumerate(mesh.cells.x):
        for iy, y in enumerate(mesh.cells.y):
            if meshed_micromagnet.mesh_coord[ix, iy] != 0:
                mask[ix, iy, :] = True
    return mask

class MmSimParams():
    """
    A container for MuMax3 simulation parameters.

    This class stores all physical and simulation-specific attributes for micromagnet simulation.
    Useful for passing around settings between magnet generation and optimization modules.

    Attributes:
        thickness (float): Magnet thickness (µm).
        saturation_magnetization (float): Saturation magnetization (A/m).
        magnetization_direction (tuple): Unit vector of initial magnetization.
        micromagnet_simulation_setting (list): Flags for [Zeeman, Uniaxial, Exchange, DM, Cubic].
        external_field (float): External magnetic field strength (mT).
        external_field_direction (tuple): Unit vector of external field direction.
        magnetocrystalline_anisotropy_constant (float): Ku in J/m³.
        uniaxial_axis (tuple): Unit vector of uniaxial anisotropy.
        exchange_constant (float): Aex in J/m.
        layername (str): Name of the target layer (used for filtering gates).
    """
    def __init__(self, 
                 thickness : float = None, #Thickness of a magnet
                 saturation_magnetization : float = None, #A/m, saturation magnetization of the material comprising a micromagnet 
                 magnetization_direction : (float, float, float) = None, #direction of the saturated magnetization 
                 micromagnet_simulation_setting = [0, 0, 0, 0, 0], #[Zeeman, Uniaxial, Exchange, DM energy, Cubic energy
                 external_field : float = None, # mT
                 external_field_direction : (float, float, float) = None, #The unit vector of the external field in the order of (x, y, z)
                 magnetocrystalline_anisotropy_constant = None, #J/m^3, magnetocrystalline anisotropy constant
                 uniaxial_axis: (float, float, float) = None, #Diection of the uniaxial axis
                 exchange_constant : float = None, #exchange Constant, J/m
                 layername : str = None):
        self.simulation_parameter = {
            'thickness' : thickness,
            'saturation_magnetization' : saturation_magnetization,
            'micromagnet_simulation_setting': micromagnet_simulation_setting,
            'magnetization_direction': magnetization_direction,
            'external_field': external_field,
            'external_field_direction': external_field_direction,
            'magnetocrystalline_anisotropy_constant': magnetocrystalline_anisotropy_constant,
            'uniaxial_axis': uniaxial_axis,
            'exchange_constant': exchange_constant,
            'layername': layername}
        

class MicroMagnet:
    """
    GPU-accelerated micromagnet simulator using MuMax3.

    This class simulates a micromagnetic object using MuMax3 via OVF files.
    It provides functionality for defining device geometry, running the simulation,
    and computing stray magnetic fields using a custom C++ kernel.

    Attributes:
        thickness_um (float): Magnet thickness in micrometers.
        Ms (float): Saturation magnetization (A/m).
        m_dir (tuple): Initial magnetization direction.
        layer (str): Name of the layer associated with this micromagnet.
        Hext (tuple or None): External magnetic field vector (A/m).
        Ku (float or None): Magnetocrystalline anisotropy constant (J/m^3).
        uniax_axis (tuple or None): Direction of uniaxial anisotropy.
        Aex (float or None): Exchange constant (J/m).
    """
    def __init__(self,
                 thickness: float,
                 saturation_magnetization: float,
                 magnetization_direction: Tuple[float, float, float],
                 micromagnet_simulation_setting: List[int] = [0, 0, 0, 0, 0],
                 external_field: float | None = None,
                 external_field_direction: Tuple[float, float, float] | None = None,
                 magnetocrystalline_anisotropy_constant: float | None = None,
                 uniaxial_axis: Tuple[float, float, float] | None = None,
                 exchange_constant: float | None = None,
                 layername: str | None = None) -> None:
        # store geometry in µm internally (compat)
        self.thickness_um = thickness * 1e6  # µm
        self.Ms = saturation_magnetization
        self.m_dir = magnetization_direction
        self.layer = layername

        self._setting = dict(
            Zeeman=micromagnet_simulation_setting[0],
            Uniaxial=micromagnet_simulation_setting[1],
            Exchange=micromagnet_simulation_setting[2],
        )
        if self._setting["Zeeman"]:
            if external_field is None or external_field_direction is None:
                raise ValueError("Zeeman term requested but field missing")
            self.Hext = tuple(c*external_field*conversionfactor_of_mT_into_Am for c in external_field_direction) #Stores mT input into A/m.
        else:
            self.Hext = None

        self.Ku = magnetocrystalline_anisotropy_constant if self._setting["Uniaxial"] else None
        self.uniax_axis = uniaxial_axis if self._setting["Uniaxial"] else None
        self.Aex = exchange_constant if self._setting["Exchange"] else None

        # placeholders
        self.magnets = []
        self.mesh = None
        self.system = None
        self.field_x = self.field_y = self.field_trace = None

    def Define(self, 
               QD, 
               x_range, 
               y_range, 
               gran, 
               layername=None):
        """
        Set geometry and field grid for the magnetic region based on gate data.

        Args:
            QD: Quantum dot device object.
            x_range (tuple): x-axis range (min, max) in meters.
            y_range (tuple): y-axis range (min, max) in meters.
            gran (tuple): (nx, ny) number of grid points.
            layername (str or None): Specific layer name to filter gates.
        """
        self.field_x = np.linspace(*x_range, gran[0]) # in meter
        self.field_y = np.linspace(*y_range, gran[1]) # in meter
        self.field_trace = np.zeros((gran[0], gran[1], 3))
        target = layername or self.layer
        self.magnets = [g for g in QD.gate_set.gate_list if g.name == target]
        xs = [g.point_list_x for g in self.magnets]; ys = [g.point_list_y for g in self.magnets]
        self.x_min, self.x_max = float(np.min(xs)), float(np.max(xs))
        self.y_min, self.y_max = float(np.min(ys)), float(np.max(ys))

    def FetchFromCAD(self,
                     QD: "Quantum_dot_device",
                     field_x_list: Tuple[float, float],
                     field_y_list: Tuple[float, float],
                     gran_list: Tuple[int, int],
                     x_list: Tuple[float, float] | None = None,
                     y_list: Tuple[float, float] | None = None,
                     layername: str | None = None) -> None:
        """
        Load geometry and grid settings from a CAD-defined quantum dot device.

        Args:
            QD: Quantum dot device.
            field_x_list (tuple): x-range of field evaluation (m).
            field_y_list (tuple): y-range of field evaluation (m).
            gran_list (tuple): Number of grid points in (x, y).
            x_list (tuple or None): Physical x-bounds of magnet.
            y_list (tuple or None): Physical y-bounds of magnet.
            layername (str or None): Target layer name.
        """
        self.Define(QD, field_x_list, field_y_list, gran_list, layername)
        if x_list is not None:
            self.magnet_x_list = list(x_list)
        if y_list is not None:
            self.magnet_y_list = list(y_list)

    def MeshCoordFetch(self, 
                       x_range: Tuple[float, float], 
                       y_range: Tuple[float, float], 
                       gran: Tuple[int, int]) -> np.ndarray:
        """
        Fetches Mesh coordinate

        Args:
            x_range (tuple): x-range.
            y_range (tuple): y-range.
            gran (tuple): granularity in (nx, ny).
        """
        nx, ny = gran
        mask = np.zeros(gran, dtype=int)
        xs = np.linspace(*x_range, nx, endpoint=False)
        ys = np.linspace(*y_range, ny, endpoint=False)
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                if any(g.gate_polygon.contains(Point(x, y)) for g in self.magnets):
                    mask[ix, iy] = 1
        return mask
    #Mumax3-based micromagnet simulator; the function name is retained as "LaunchOOMMFC" for the compatibility.
    def LaunchOOMMFC(self, 
                     n, 
                     mumax_bin="mumax3"):
        """
        Run MuMax3 simulation to compute equilibrium magnetization.

        Uses temporary files to generate OVF inputs and scripts,
        and reads back simulation output from MuMax3.

        Args:
            n (tuple): Grid size (nx, ny, nz).
            mumax_bin (str): Path to MuMax3 executable. If mumax3 is install, use the default variable.
        """
        _start = time.time()
        # Build mesh (m units for MuMax3)
        p1 = (self.x_min*1e-6, self.y_min*1e-6, 0)
        p2 = (self.x_max*1e-6, self.y_max*1e-6, self.thickness_um*1e-6)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
        mask = _make_mask(mesh, self.magnets, self.x_min, self.y_min)

        # initial field
        arr = np.zeros((*n, 3))
        for i in range(3):
            arr[..., i] = self.m_dir[i] * mask * self.Ms
        init_field = df.Field(mesh, nvdim=3, value=arr, valid=mask)

        with tempfile.TemporaryDirectory() as tmp:
            
            init_ovf = os.path.join(tmp, "init.ovf"); init_field.to_file(init_ovf)
            script_path = os.path.join(tmp, "sim.mx3")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(self._mx3_script(n, init_ovf))
            #subprocess.run(
            #    ["mumax3", "-remote", _mumax_server_address, script_path],
            #    check=True,
            #    cwd=tmp,
            #    stdout=subprocess.DEVNULL
            #)
            subprocess.run([mumax_bin,"-s", "-http", ":0",script_path], 
                           check=True, 
                           cwd=tmp,
                           stdout=subprocess.DEVNULL) # Commented out due to the kernel termination problem.
            
            # read result (first OVF in sim.out)
            ovf_dir = os.path.join(tmp, "sim.out")
            ovf_file = next(p for p in os.listdir(ovf_dir) if p.endswith(".ovf"))
            m_field = df.Field.from_file(os.path.join(ovf_dir, ovf_file))
            #-------------------runner code-start---------------------------------
            #init_ovf = Path(tmp)/"init.ovf"; init_field.to_file(init_ovf)
            #script_text = self._mx3_script(n, init_ovf)
            #ovf_path = submit(script_text) 
            #m_field  = df.Field.from_file(ovf_path)
            #-------------------runner code-end-----------------------------------
            # shift the origin; mumax enforces (0,0,0) to be an origin
            shift_x = self.x_min * 1e-6          # m
            shift_y = self.y_min * 1e-6
            reg_shift = m_field.mesh.region.translate((shift_x, shift_y, 0))
            mesh_shift = df.Mesh(region=reg_shift, n=m_field.mesh.n)
            m_field = df.Field(mesh_shift, nvdim=3, value=m_field.array)
            # re‑apply mask to be safe
            arr = m_field.array; arr[~mask] = 0
            self.system = SimpleNamespace(m=df.Field(m_field.mesh, nvdim=3, value=arr, valid=mask))
            self.mesh = m_field.mesh
        print('Total time elapsed:', datetime.timedelta(seconds=time.time() - _start))

    # mumax3 script builder
    def _mx3_script(self, 
                    n, 
                    init_ovf):
        """
        Generate a MuMax3 script string for the current simulation.

        Args:
            n (tuple): Grid resolution.
            init_ovf (str): Path to initial OVF magnetization file.

        Returns:
            str: MuMax3 script text.
        """
        nx, ny, nz = n
        dx = (self.x_max-self.x_min)/nx/1e6
        dy = (self.y_max-self.y_min)/ny/1e6
        dz = self.thickness_um/1e6/nz
        lines = [
            f"SetGridsize({nx},{ny},{nz})",
            f"SetCellsize({dx},{dy},{dz})",
            f"Msat={self.Ms}",
            f"m.LoadFile(\"{init_ovf}\")",  # zero outside mask
        ]
        if self.Aex is not None:
            lines.append(f"Aex={self.Aex}")
        if self.Ku is not None and self.uniax_axis is not None:
            ux,uy,uz=self.uniax_axis; lines += [f"Ku1={self.Ku}", f"anisU=vector({ux},{uy},{uz})"]
        if self.Hext is not None:
            bx,by,bz=(c*mu_0 for c in self.Hext); lines.append(f"B_ext=vector({bx},{by},{bz})") #converts A/m into T.
        lines += ["Minimize()", "Save(m)"]
        return "\n".join(lines)
 
    #Stray-field calculator; C++ kernel is used.
    def CalcStray(self, 
                  z_pos, 
                  component=1, 
                  n_cpu=-1):
        """
        Compute the stray magnetic field at a fixed z position.

        Uses a compiled C++ kernel to efficiently calculate the field from the magnet volume.

        Args:
            z_pos (float): z-position of evaluation plane (m).
            component (int): Field component to compute (0=x, 1=y, 2=z).
            n_cpu (int): Number of threads to use (-1 for auto).
        """
        if n_cpu < 1:
            n_cpu = os.cpu_count()
        elif n_cpu >= os.cpu_count():
            raise Exception('The number of cpu (ncpu) for multiprocessing exceeds the number of possible cores')
    
        m_field = self.system.m
        m_arr   = m_field.array * m_field.mesh.dV * self.Ms
        pos     = m_field.mesh.cells
        # Contiguous array generation for a C++ implementation.
        cx = np.ascontiguousarray(pos.x,        dtype=np.float64)
        cy = np.ascontiguousarray(pos.y,        dtype=np.float64)
        cz = np.ascontiguousarray(pos.z,        dtype=np.float64)
        xf = np.ascontiguousarray(self.field_x, dtype=np.float64)
        yf = np.ascontiguousarray(self.field_y, dtype=np.float64)
        mx = np.ascontiguousarray(m_arr[...,0], dtype=np.float64)
        my = np.ascontiguousarray(m_arr[...,1], dtype=np.float64)
        mz = np.ascontiguousarray(m_arr[...,2], dtype=np.float64)
    
        answer = np.zeros((self.field_x.size, self.field_y.size), dtype=np.float64)
    
        _cxx_magcalc.straycalc(
            answer, cx, cy, cz, xf, yf, mx, my, mz,
            float(z_pos), int(component), int(n_cpu)   
        )
    
        self.field_trace[:, :, component] = answer


class MagnetMesh():
    """
    Defines a rectangular magnetic mesh using a binary layout.

    Attributes:
        mesh_coord (np.ndarray): 2D array representing the magnet presence (0 or 1).
        mesh_x_boundary (tuple): (x_min, x_max) boundary of the mesh (µm).
        mesh_y_boundary (tuple): (y_min, y_max) boundary of the mesh (µm).
        mesh_x_list (np.ndarray): x-axis mesh cell boundaries.
        mesh_y_list (np.ndarray): y-axis mesh cell boundaries.
    """
    def __init__(self,
                 mesh_coord, #Mesh status matrix. Either 0 or 1 and 2d numpy array (n_x, n_y)
                 x_list : [float, float], #[x_min, x_max] for MagnetMesh, unit: um
                 y_list : [float, float], #[y_min, y_max] for MagnetMesh, unit: um
                 ):
        self.mesh_x_boundary = x_list
        self.mesh_y_boundary = y_list
        self.mesh_x_list = np.linspace(*x_list, np.shape(mesh_coord)[0]+1)
        self.mesh_y_list = np.linspace(*y_list, np.shape(mesh_coord)[1]+1)
        self.mesh_coord = mesh_coord
    
class MeshedMicroMagnet(MicroMagnet):
    """
    A MicroMagnet subclass initialized using a predefined binary mesh.

    Attributes:
        magnet_mesh (MagnetMesh): Magnet layout as a binary mesh.
    """
    def __init__(self, 
                 magnetmesh, #MagnetMesh
                 thickness : float, #Thickness of a magnet
                 saturation_magnetization : float, #A/m, saturation magnetization of the material comprising a micromagnet 
                 magnetization_direction : (float, float, float), #direction of the saturated magnetization 
                 micromagnet_simulation_setting = [0, 0, 0, 0, 0], #[Zeeman, Uniaxial, Exchange, DM energy, Cubic energy
                 external_field : float = None, # mT
                 external_field_direction : (float, float, float) = None, #The unit vector of the external field in the order of (x, y, z)
                 magnetocrystalline_anisotropy_constant = None, #J/m^3, magnetocrystalline anisotropy constant
                 uniaxial_axis: (float, float, float) = None, #Diection of the uniaxial axis
                 exchange_constant : float = None, #exchange Constant, J/m
                 layername = None):
        super().__init__(thickness,
                         saturation_magnetization,
                         magnetization_direction,
                         micromagnet_simulation_setting,
                         external_field,
                         external_field_direction,
                         magnetocrystalline_anisotropy_constant,
                         uniaxial_axis,
                         exchange_constant)
        self.magnet_mesh = magnetmesh
        
    def SystemGen(self, 
                  z_gran: int, 
                  n_cpu: int = 4,
                  mumax_bin="mumax3"):
        """
        Generate the micromagnetic system using the mesh geometry and run MuMax3 simulation.

        Args:
            z_gran (int): Number of layers along the z-direction.
            n_cpu (int): Number of CPU threads.
            mumax_bin (str): Path to MuMax3 executable.
        """
        _start = time.time()
        _p1 = (self.magnet_mesh.mesh_x_boundary[0]*1e-6, self.magnet_mesh.mesh_y_boundary[0]*1e-6, 0) 
        _p2 = (self.magnet_mesh.mesh_x_boundary[1]*1e-6, self.magnet_mesh.mesh_y_boundary[1]*1e-6, self.thickness_um*1e-6)
        _n = (*np.shape(self.magnet_mesh.mesh_coord), z_gran)
        _mesh = df.Mesh(region=df.Region(p1=_p1, p2=_p2), n=_n)
        self.x_min, self.x_max = float(np.min(self.magnet_mesh.mesh_x_list)), float(np.max(self.magnet_mesh.mesh_x_list))
        self.y_min, self.y_max = float(np.min(self.magnet_mesh.mesh_y_list)), float(np.max(self.magnet_mesh.mesh_y_list))
        _mask = _make_mask_mesh(_mesh, self.magnet_mesh, self.x_min, self.y_min)
        m_array = np.zeros((*_n, 3))
        for i in range(3):
            m_array[..., i] = self.m_dir[i] * _mask * self.Ms
        init_field = df.Field(_mesh, nvdim=3, value=m_array, valid=_mask)
        
        with tempfile.TemporaryDirectory() as tmp:
            init_ovf = os.path.join(tmp, "init.ovf"); init_field.to_file(init_ovf)
            script_path = os.path.join(tmp, "sim.mx3")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(self._mx3_script(_n, init_ovf))
            subprocess.run([mumax_bin,"-s", "-http", ":0",script_path], 
                           check=True, 
                           cwd=tmp,
                           stdout=subprocess.DEVNULL) # Commented out due to the kernel termination problem.
            #subprocess.run([mumax_bin, script_path], 
            #               check=True, 
            #               cwd=tmp,
            #               stdout=subprocess.DEVNULL)
            # read result (first OVF in sim.out)
            ovf_dir = os.path.join(tmp, "sim.out")
            ovf_file = next(p for p in os.listdir(ovf_dir) if p.endswith(".ovf"))
            m_field = df.Field.from_file(os.path.join(ovf_dir, ovf_file))
            #-------------------runner code-start---------------------------------
            #init_ovf = os.path.join(tmp, "init.ovf"); init_field.to_file(init_ovf)
            #script_text = self._mx3_script(_n, init_ovf)
            #ovf_path = submit(script_text, timeout = 600)
            #m_field  = df.Field.from_file(ovf_path)
            #-------------------runner code-end-----------------------------------
            # shift the origin; mumax enforces (0,0,0) to be an origin
            shift_x = self.x_min * 1e-6          # m
            shift_y = self.y_min * 1e-6
            reg_shift = m_field.mesh.region.translate((shift_x, shift_y, 0))
            mesh_shift = df.Mesh(region=reg_shift, n=m_field.mesh.n)
            m_field = df.Field(mesh_shift, nvdim=3, value=m_field.array)
            # re‑apply mask to be safe
            arr = m_field.array; arr[~_mask] = 0
            self.system = SimpleNamespace(m=df.Field(m_field.mesh, nvdim=3, value=arr, valid=_mask))
            self.mesh = m_field.mesh
        print('Total time elapsed:', datetime.timedelta(seconds=time.time() - _start))

    def CalcStray(self,
                  x_range: Tuple[float, float],#list of x coordinates within which stray field is calculated, [min, max]
                  y_range: Tuple[float, float],#list of y coordinates within which stray field is calculated, [min, max]
                  gran: Tuple[int, int], #granularity for x_list and y_list
                  quantum_dot_position_z: float,
                  component: int = 1, #component to calc (0 = x, 1 = y, 2 = z)
                  n_cpu: int = -1, #of threads to use
                  ):
        """
        Calculate the stray field at a given z-position using the generated micromagnet system.

        Args:
            x_range (tuple): Range for x-coordinate (min, max).
            y_range (tuple): Range for y-coordinate (min, max).
            gran (tuple): Number of grid points along x and y.
            quantum_dot_position_z (float): z-position of the evaluation plane.
            component (int): Component of the magnetic field to evaluate (0=x, 1=y, 2=z).
            n_cpu (int): Number of CPU threads to use.
        """
        self.field_x = np.linspace(x_range[0], x_range[1], gran[0])
        self.field_y = np.linspace(y_range[0], y_range[1], gran[1])
        if self.field_trace is None:
            self.field_trace = np.zeros((len(self.field_x), len(self.field_y), 3)) 
        super().CalcStray(quantum_dot_position_z, component, n_cpu)
