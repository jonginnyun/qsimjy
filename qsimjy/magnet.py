import multiprocess as mp
import time
import datetime
import discretisedfield as df
import oommfc as oc
import os
import micromagneticmodel as mm
import numpy as np
import _cxx_magcalc
from shapely.geometry import Polygon, MultiPolygon, Point

# Scientific Constants
conversionfactor_of_mT_into_Am = 0.79577471545947673925*1e3 #ref;https://maurermagnetic.com/en/demagnetizing/technology/convert-magnetic-units/
mu_0 = 1.25637e-6  # vacuum permeability [N * A^-2]
prop_const = mu_0 / np.pi / 4  # proportionality constant for magnetic field

class MmSimParams():
    """
    A class that generates an object containing the parameters for the class MicroMagnet.
    Added soleley for the convenience.
    Dependency: Micromagnet, MeshedMicroMagnet
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
        
class MicroMagnet():
    def __init__(self, 
                 thickness : float, #Thickness of a magnet
                 saturation_magnetization : float, #A/m, saturation magnetization of the material comprising a micromagnet 
                 magnetization_direction : (float, float, float), #direction of the saturated magnetization 
                 micromagnet_simulation_setting = [0, 0, 0, 0, 0], #[Zeeman, Uniaxial, Exchange, DM energy, Cubic energy
                 external_field : float = None, # mT
                 external_field_direction : (float, float, float) = None, #The unit vector of the external field in the order of (x, y, z)
                 magnetocrystalline_anisotropy_constant = None, #J/m^3, magnetocrystalline anisotropy constant
                 uniaxial_axis: (float, float, float) = None, #Diection of the uniaxial axis
                 exchange_constant : float = None, #exchange Constant, J/m
                 layername : str = None):
        self.magnets = []
        self.magnet_layer_name = layername
        self.magnet_x_list = None
        self.magnet_y_list = None
        self.thickness = thickness
        self.saturation_magnetization = saturation_magnetization
        self.magnetization_direction = magnetization_direction
        self._setting = {"Zeeman" : micromagnet_simulation_setting[0], 
                         "Uniaxial" : micromagnet_simulation_setting[1], 
                         "Exchange": micromagnet_simulation_setting[2], 
                         "DM energy" : micromagnet_simulation_setting[3], 
                         "Cubic Energy" : micromagnet_simulation_setting[4]}

        self.external_field = None
        self.exchange_constant = None
        self.magnetocrystalline_anisotropy_constant = None
        self.uniaxial_axis = None
        self.system = None
        self.field_x = None
        self.field_y = None
        self.field_trace = None
        
        if self._setting["Zeeman"] == 1:
            if ((external_field == None) or (external_field_direction == None)):
                raise Exception('input external field value. Required variables are external_field (mT, magnitude) and external_field_direction (unit vector).')
            else:
                self.external_field = tuple([i*external_field*conversionfactor_of_mT_into_Am for i in external_field_direction])
        if self._setting["Uniaxial"] == 1:
            if ((uniaxial_axis == None) or (magnetocrystalline_anisotropy_constant == None)):
                raise Exception('input Uniaxial anisotropy value. Required variables are magnetocrystalline_anisotropy_constant (J/m^3, magnitude) and uniaxial_axis (axis direction).')
            else:
                self.uniaxial_axis = uniaxial_axis
                self.magnetocrystalline_anisotropy_constant = magnetocrystalline_anisotropy_constant
        if self._setting["Exchange"] == 1:
            if ((exchange_constant == None)):
                raise Exception('input exchange value. Required variable is exchange_constant (J/m)')
            else:
                self.exchange_constant = exchange_constant
        #Additional Settings are to be added
    
    def Define(self, 
               QD: 'Quantum_dot_device', 
               x_list: [float, float],
               y_list: [float, float],
               gran_list: [int, int],
               layername = None):
        self.field_x = np.linspace(x_list[0], x_list[1], gran_list[0])
        self.field_y = np.linspace(y_list[0], y_list[1], gran_list[1])
        self.field_trace = np.zeros((len(self.field_x), len(self.field_y), 3))  # Field array to store B value at each position
        
        if layername != None:
            for i in QD.gate_set.gate_list:
                if (i.name == layername):
                    self.magnets.append(i)
        else:
            for i in QD.gate_set.gate_list:
                if (i.name == self.magnet_layer_name):
                    self.magnets.append(i)
        _list_x = [i.point_list_x for i in self.magnets]
        _list_y = [i.point_list_y for i in self.magnets]
        self.magnet_x_list = [np.min(_list_x), np.max(_list_x)]
        self.magnet_y_list = [np.min(_list_y), np.max(_list_y)]
        
    def FetchFromCAD(self, 
                     QD: 'Quantum_dot_device',  
                     field_x_list: [float, float], 
                     field_y_list: [float, float],
                     gran_list: [int, int],
                     x_list = None,
                     y_list = None,
                     layername = None):
        self.field_x = np.linspace(x_list[0], x_list[1], gran_list[0])
        self.field_y = np.linspace(y_list[0], y_list[1], gran_list[1])
        self.field_trace = np.zeros((len(self.field_x), len(self.field_y), 3))  # Field array to store B value at each position
        
        if layername != None:
            for i in QD.gate_set.gate_list:
                if (i.name == layername):
                    self.magnets.append(i)
        else:
            for i in QD.gate_set.gate_list:
                if (i.name == self.magnet_layer_name):
                    self.magnets.append(i)
        _list_x = [i.point_list_x for i in self.magnets]
        _list_y = [i.point_list_y for i in self.magnets]
        if x_list != None:
            self.magnet_x_list = x_list
        if y_list != None:
            self.magnet_y_list = y_list

    def LaunchOOMMFC(self, n, n_cpu = 4):
        def _Ms_value(pos):
            x, y, z = pos
            if ((z>=0) and (z<=self.thickness)):
                for i in self.magnets:
                    if i.gate_polygon.contains(Point(x*1e6,y*1e6)):
                        return self.saturation_magnetization
                return 0
            return 0
        def _ms_value(pos):
            x, y, z = pos
            if ((z>=0) and (z<=self.thickness)):
                for i in self.magnets:
                    if i.gate_polygon.contains(Point(x*1e6,y*1e6)):
                        return self.saturation_magnetization 
                return (0, 0, 0)
            return (0, 0, 0)
        _x_min, _x_max = self.magnet_x_list[0], self.magnet_x_list[1]
        _y_min, _y_max = self.magnet_y_list[0], self.magnet_y_list[1]
        _p1 = (_x_min*1e-6, _y_min*1e-6, 0) 
        _p2 = (_x_max*1e-6, _y_max*1e-6, self.thickness)
        self.region = df.Region(p1=_p1, p2=_p2)
        self.mesh = df.Mesh(region = self.region, n = n)
        print('Meshes generated. Configuring systems ...')
        self.system = mm.System(name = "micromagnet")
        _all_null = 1
        self.system.energy = mm.Demag()
        if self._setting["Zeeman"] == 1 : 
            self.system.energy += mm.Zeeman(H=self.external_field)
        if self._setting["Uniaxial"] == 1:
            self.system.energy += mm.UniaxialAnisotropy(K=self.magnetocrystalline_anisotropy_constant,
                                                        u=self.uniaxial_axis)
        if self._setting["Exchange"] == 1: 
            self.system.energy += mm.Exchange(A = self.exchange_constant)
        self.system.m=df.Field(self.mesh, nvdim = 3, value = (0, 1, 0), norm = _Ms_value, valid = "norm")
        md = oc.MinDriver()  # create energy minimisation driver
        print('System configured.')
        md.drive(self.system, n_threads = n_cpu)  # run energy minimisation
        
    def CalcStray(self,
                  quantum_dot_position_z, 
                  component = 1,
                  n_cpu = 4):
        start = time.time()
        # Checking the validity of the number of CPUs
        
        if n_cpu == -1:
            n_cpu = os.cpu_count()
        elif n_cpu >= os.cpu_count():
            raise Exception('The number of cpu (ncpu) for multiprocessing exceeds the number of possible cores')
        
    
        m_field = self.system.m
        m_array = m_field.array
        pos_obj = m_field.mesh.cells
        pos_array = np.zeros((len(pos_obj.x),len(pos_obj.y), len(pos_obj.z), 3))
        for i in range(len(pos_obj.x)):
            for j in range(len(pos_obj.y)):
                for k in range(len(pos_obj.z)):
                    pos_array[i, j, k, 0] = pos_obj.x[i]
                    pos_array[i, j, k, 1] = pos_obj.y[j]
                    pos_array[i, j, k, 2] = pos_obj.z[k]
        cell_volume = m_field.mesh.dV
        
        
        # Assuming 'pos_obj' and 'system' are defined
        m_field = self.system.m
        pos_obj = m_field.mesh.cells
        nx_int, ny_int, nz_int = len(pos_obj.x), len(pos_obj.y), len(pos_obj.z)
        n_array = [nx_int, ny_int, nz_int]
        
        m_array = m_field.array * m_field.mesh.dV  # Shape: (nx_int, ny_int, nz_int, 3)
        # Contiguous array generation for a C++ implementation
        _mx_array_contiguous = np.ascontiguousarray(m_array[:, :, :, 0], dtype = np.float64)
        _my_array_contiguous = np.ascontiguousarray(m_array[:, :, :, 1], dtype = np.float64)
        _mz_array_contiguous = np.ascontiguousarray(m_array[:, :, :, 2], dtype = np.float64)
        _x_fine_contiguous = np.ascontiguousarray(self.field_x, dtype = np.float64)
        _y_fine_contiguous = np.ascontiguousarray(self.field_y, dtype = np.float64)
        answer_trace = np.ascontiguousarray(np.zeros((np.shape(self.field_x)[0], np.shape(self.field_y)[0])))
        # Invoking c bound stray field calculating function
        _cxx_magcalc.straycalc(answer_trace, 
                               pos_obj.x, 
                               pos_obj.y,
                               pos_obj.z,
                               _x_fine_contiguous,
                               _y_fine_contiguous,
                               _mx_array_contiguous,
                               _my_array_contiguous,
                               _mz_array_contiguous,
                               quantum_dot_position_z,
                               component,
                               n_cpu)
        end = time.time()
        self.field_trace[:, :, component] = answer_trace
        print('Total time elapsed:', datetime.timedelta(seconds=end - start))
        
    def MeshCoordFetch(self, 
                       x_list : [float, float], #[x_min, x_max] for MagnetMesh, unit: um
                       y_list : [float, float], #[y_min, y_max] for MagnetMesh, unit: um
                       gran_list : [int, int] #Granularity of the mesh whose values to be generated
                       ):
        _mesh_to_return = np.zeros(gran_list)
        _mesh_x_list = np.linspace(*x_list, gran_list[0]+1)
        _mesh_y_list = np.linspace(*y_list, gran_list[1]+1)
        for i in range(gran_list[0]*gran_list[1]):
            ix = i // gran_list[1]
            iy = i % gran_list[1]
            for j in self.magnets:
                if j.gate_polygon.contains(Point(_mesh_x_list[ix]*0.5+_mesh_x_list[ix+1]*0.5,
                                                 _mesh_y_list[iy]*0.5+_mesh_y_list[iy+1]*0.5)):
                    _mesh_to_return[ix, iy] = 1
                    break
        return _mesh_to_return
        


class MeshedMicroMagnet(MicroMagnet): #Micromagnet Class to Handle the Mesh
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

    def SystemGen(self, z_gran, n_cpu = 4):
        _Ms_array = np.repeat(self.magnet_mesh.mesh_coord[:, :, None], z_gran , axis=2)
        _p1 = (self.magnet_mesh.mesh_x_boundary[0]*1e-6, self.magnet_mesh.mesh_y_boundary[0]*1e-6, 0) 
        _p2 = (self.magnet_mesh.mesh_x_boundary[1]*1e-6, self.magnet_mesh.mesh_y_boundary[1]*1e-6, self.thickness)
        self.region = df.Region(p1=_p1, p2=_p2)
        self.Mesh = df.Mesh(region = self.region, n = (*np.shape(self.magnet_mesh.mesh_coord), z_gran))
        print('Ubermag meshes generated. Configuring systems ...')
        self.system = mm.System(name = "micromagnet")
        self.system.energy = mm.Demag()
        if self._setting["Zeeman"] == 1 : 
            self.system.energy += mm.Zeeman(H=self.external_field)
        if self._setting["Uniaxial"] == 1:
            self.system.energy += mm.UniaxialAnisotropy(K=self.magnetocrystalline_anisotropy_constant,
                                                        u=self.uniaxial_axis)
        if self._setting["Exchange"] == 1: 
            self.system.energy += mm.Exchange(A = self.exchange_constant)
        self.system.m=df.Field(self.Mesh, nvdim = 3, value = (0, 1, 0), norm = _Ms_array*self.saturation_magnetization, valid = "norm")
        md = oc.MinDriver()  # create energy minimisation driver
        print('System configured.')
        md.drive(self.system, n_threads = n_cpu)
    
    def CalcStray(self,
                  x_list: [float, float], #list of x coordinates within which stray field is calculated, [min, max]
                  y_list: [float, float], #list of y coordinates within which stray field is calculated, [min, max]
                  gran_list: [int, int], #granularity for x_list and y_list
                  quantum_dot_position_z,
                  component = 1, #component to calc (0 = x, 1 = y, 2 = z)
                  n_cpu = 4): #of threads to use
        self.field_x = np.linspace(x_list[0], x_list[1], gran_list[0])
        self.field_y = np.linspace(y_list[0], y_list[1], gran_list[1])
        if self.field_trace is None:
            self.field_trace = np.zeros((len(self.field_x), len(self.field_y), 3)) 
        super().CalcStray(quantum_dot_position_z, component, n_cpu)
    
class MagnetMesh():
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