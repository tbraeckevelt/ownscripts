import torch
import numpy as np
import h5py
import yaff
import molmod

import ase.units
from ase.io import read, write
from ase.stress import voigt_6_to_full_3x3_stress
from ase.cell import Cell
from ase import Atoms
from pathlib import Path
from mace.calculators import MACECalculator

class ForcePartASE(yaff.pes.ForcePart):
    """YAFF Wrapper around an ASE calculator"""

    def __init__(self, system, atoms, calculator):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            system object

        atoms : ase.Atoms
            atoms object with calculator included.

        """
        yaff.pes.ForcePart.__init__(self, 'ase', system)
        self.system = system # store system to obtain current pos and box
        self.atoms  = atoms
        self.calculator = calculator

    def _internal_compute(self, gpos=None, vtens=None):
        self.atoms.set_positions(self.system.pos / molmod.units.angstrom)
        self.atoms.set_cell(Cell(self.system.cell._get_rvecs() / molmod.units.angstrom))
        energy = self.atoms.get_potential_energy() * molmod.units.electronvolt
        if gpos is not None:
            forces = self.atoms.get_forces()
            gpos[:] = -forces * molmod.units.electronvolt / molmod.units.angstrom
        if vtens is not None:
            volume = np.linalg.det(self.atoms.get_cell())
            stress = voigt_6_to_full_3x3_stress(self.atoms.get_stress())
            vtens[:] = volume * stress * molmod.units.electronvolt
        return energy
    

def create_forcefield(atoms, calculator):
    """Creates force field from ASE atoms instance"""
    system = yaff.System(
            numbers=atoms.get_atomic_numbers(),
            pos=atoms.get_positions() * molmod.units.angstrom,
            rvecs=atoms.get_cell() * molmod.units.angstrom,
            )
    system.set_standard_masses()

    part_ase = ForcePartASE(system, atoms, calculator)
    ff = yaff.pes.ForceField(system, [part_ase])

    return ff

class ExtXYZHook(yaff.sampling.iterative.Hook):

    def __init__(self, path_xyz, step=1, start=0, append = False, write_vel = True):
        super().__init__(step=step, start=start)
        if Path(path_xyz).exists() and (append == False):
            Path(path_xyz).unlink() # remove if exists
        self.path_xyz = path_xyz
        self.atoms = None
        self.write_vel = write_vel

    def init(self, iterative):
        self.atoms = Atoms(
                numbers=iterative.ff.system.numbers.copy(),
                positions=iterative.ff.system.pos / molmod.units.angstrom,
                cell=iterative.ff.system.cell._get_rvecs() / molmod.units.angstrom,
                pbc=True,
                )

    def pre(self, iterative):
        pass

    def post(self, iterative):
        pass

    def __call__(self, iterative):
        if self.atoms is None:
            self.init(iterative)
        self.atoms.set_positions(iterative.ff.system.pos / molmod.units.angstrom)
        cell = iterative.ff.system.cell._get_rvecs() / molmod.units.angstrom
        self.atoms.set_cell(cell)
        self.atoms.arrays['forces'] = -iterative.ff.gpos * molmod.units.angstrom / molmod.units.electronvolt
        self.atoms.info['energy'] = iterative.ff.energy / molmod.units.electronvolt
        volume = np.linalg.det(cell)
        self.atoms.info['stress'] = iterative.ff.vtens / (molmod.units.electronvolt * volume)
        if self.write_vel:
            self.atoms.set_velocities(iterative.vel * molmod.units.femtosecond / (ase.units.fs * molmod.units.angstrom) )
        write(self.path_xyz, self.atoms, parallel=False, append=True)

def get_calculator(path_calc, device, dtype_str):
    if path_calc[-6:] == ".model":
        calculator = MACECalculator(path_calc, device=device, default_dtype = dtype_str)
        if dtype_str == 'float64':
            calculator.model.double()                    #This is because MACE has a bug in converting the attributes!
    else:
        raise NotImplementedError()
    return calculator

def get_timestep(atoms):
    #Very crude, but very safe heuristic to set the timestep
    mass_min = min(atoms.get_masses())
    if mass_min < 5.0:
        return 0.5 #in femteseconds
    elif mass_min < 50.0:
        return 1.0 #in femteseconds
    else:
        return 2.0 #in femteseconds


def run_MD(inputs=[], outputs=[], steps = 100, step = 50, temperature = None, pressure = None, seed = 0, num_cores = 1, precision = "single", device = "cuda"):
    import torch
    import numpy as np
    import h5py
    import yaff
    import molmod
    from ase.io import read, write

    path_atoms = str(inputs[0]) 
    path_calc  = str(inputs[1])
    path_traj  = str(outputs[0])
    assert path_atoms[-4:] == ".xyz", "The first input must be a .xyz file"
    assert path_calc[-6:] == ".model", "The second input must be a .model file"
    np.random.seed(seed)
    torch.set_num_threads(num_cores)

    atoms =read(path_atoms)
    timestep = get_timestep(atoms)

    dtype_str = 'float32'
    if precision == 'double':
        torch.set_default_dtype(torch.float64)
        dtype_str = 'float64'
    calculator = get_calculator(path_calc, device, dtype_str)
    atoms.calc = calculator

    # create forcefield from atoms
    ff = create_forcefield(atoms, calculator)

    #Hooks
    hooks = []
    if path_traj[-3:] == '.h5':
        h5file = h5py.File(path_traj, 'w')
        hooks.append(yaff.HDF5Writer(h5file, step=step, start=0))
    elif path_traj[-4:] == '.xyz':
        hooks.append(ExtXYZHook(path_traj, start=0, step=step))
    hooks.append(yaff.VerletScreenLog(step=step, start=0))

    # temperature / pressure control
    if temperature is None:
        print('CONSTANT ENERGY, CONSTANT VOLUME')
    else:
        thermo = yaff.LangevinThermostat(temperature, timecon=100 * molmod.units.femtosecond)
        hooks.append(thermo)
        if pressure is None:
            print('CONSTANT TEMPERATURE, CONSTANT VOLUME')
        else:
            print('CONSTANT TEMPERATURE, CONSTANT PRESSURE')
            vol_constraint = False
            print('Langevin barostat')
            baro = yaff.LangevinBarostat(
                    ff,
                    temperature,
                    pressure,
                    timecon=molmod.units.picosecond,
                    anisotropic=True,
                    vol_constraint=vol_constraint,
                    )
            tbc = yaff.TBCombination(thermo, baro)
            hooks.append(tbc)

    # integration
    verlet = yaff.VerletIntegrator(
            ff,
            timestep=timestep*molmod.units.femtosecond,
            hooks=hooks,
            temp0=temperature, # initialize velocities to correct temperature, if vel0 is None
            )
    yaff.log.set_level(yaff.log.medium)
    verlet.run(steps)
    yaff.log.set_level(yaff.log.silent)
