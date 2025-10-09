from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmplumed import PlumedForce
from network import Network
import numpy as np
import os, shutil
import tensorflow as tf
from datetime import datetime

def main():

    # setup simulation
    simulation, system, gro, top, ml_force, output_dir = setup()

    # run simulation
    startTime = datetime.now()
    simulate(simulation, system, gro, top, ml_force, output_dir)
    print(datetime.now() - startTime)


def setup():
    print("Setting up MD Simulation...")
    input_dir = "md_input"
    output_dir = "md_data"
    isExist = os.path.exists(output_dir)
    if isExist:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    gro = GromacsGroFile(f"{input_dir}/input.gro")
    top = GromacsTopFile(f"{input_dir}/input.top", periodicBoxVectors=gro.getPeriodicBoxVectors())

    # for rigid water to be found the water residue name must be "HOH"
    system = top.createSystem(nonbondedMethod=CutoffPeriodic, nonbondedCutoff=1*nanometer,
        constraints=None, removeCMMotion=True, rigidWater=True, switchDistance=None)

    residues = list(top.topology.residues())
    ligand_n_atom = len(list(residues[0].atoms()))

    nb = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]

    # set exceptions for all ligand atoms
    for i in range(ligand_n_atom):
        for j in range(i):
            nb.addException(i, j, 0, 1, 0)

    # create custom force for PairNet predictions
    ml_force = CustomExternalForce("-fx*x-fy*y-fz*z")
    system.addForce(ml_force)
    ml_force.addPerParticleParameter("fx")
    ml_force.addPerParticleParameter("fy")
    ml_force.addPerParticleParameter("fz")
    for j in range(ligand_n_atom):
        ml_force.addParticle(j, (0, 0, 0))

    # define ensemble, thermostat and integrator
    system.addForce(MonteCarloBarostat(1.0*bar, 298*kelvin))
    integrator = DrudeLangevinIntegrator(298*kelvin, 5 / picosecond,
        1.0 * kelvin, 20 / picosecond, 0.001 * picoseconds)

    integrator.setMaxDrudeDistance(0.02)
    # add Drude particles
    drude_force = DrudeForce()
    for atom in top.topology.atoms():
        if atom.name == "DW":
            drude_force.addParticle(atom.index, atom.index - 3, -1, -1, -1,
                Quantity(value=-1.3828, unit=elementary_charge),
                Quantity(value=0.000635, unit=nanometer ** 3), 0.0, 0.0)
    system.addForce(drude_force)

    # define biasing potentials
    plumed_file = open(f"{input_dir}/plumed.dat", "r")
    plumed_script = plumed_file.read()
    system.addForce(PlumedForce(plumed_script))
    plumed_file.close()

    # set up simulation
    simulation = Simulation(top.topology, system, integrator)
    simulation.context.setPositions(gro.positions)

    # select initial velocities from MB distribution
    simulation.context.setVelocitiesToTemperature(298*kelvin)

    return simulation, system, gro, top, ml_force, output_dir


def simulate(simulation, system, gro, top, ml_force, output_dir):
    print("Performing MD Simulation...")
    max_steps = 1000
    print_trj = 10
    print_data = 10
    print_summary = 10
    charge_scaling = 1.3
    tot_n_atom = len(gro.getPositions())

    charges = np.zeros(tot_n_atom)
    nb = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]

    for i in range(system.getNumParticles()):
        charge, sigma, epsilon = nb.getParticleParameters(i)
        charges[i] = charge.value_in_unit(elementary_charge)

    # need to get ligand_n_atom from residue instead
    residues = list(top.topology.residues())
    ligand_n_atom = len(list(residues[0].atoms()))

    tf.config.set_visible_devices([], 'GPU')

    mol = Molecule()
    network = Network(mol)
    print("Loading a trained model...")
    input_dir = "trained_model"
    ligand_atoms = np.loadtxt(f"{input_dir}/nuclear_charges.txt", dtype=np.float32).reshape(-1)
    mol.n_atom = len(ligand_atoms)
    model = network.load(mol, input_dir)

    simulation.reporters.append(StateDataReporter(f"./{output_dir}/openmm.csv",
        reportInterval=print_summary,step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, temperature=True, separator=" "))

    # this prevents tensorflow printing warnings or other information
    tf.get_logger().setLevel('ERROR')

    f1 = open(f"./{output_dir}/coords.txt", 'w')
    f2 = open(f"./{output_dir}/forces.txt", 'w')
    f3 = open(f"./{output_dir}/velocities.txt", 'w')
    f4 = open(f"./{output_dir}/energies.txt", 'w')
    f5 = open(f"./{output_dir}/charges.txt", 'w')
    f6 = open(f"./{output_dir}/ML_forces.txt", 'w')
    f7 = open(f"./{output_dir}/MM_forces.txt", 'w')

    # run MD simulation for requested number of timesteps
    print("Performing MD simulation...")
    state = simulation.context.getState(getEnergy=True)
    PE = state.getPotentialEnergy() / kilocalories_per_mole
    print("Initial Potential Energy: ", PE, "kcal/mol")

    for i in range(max_steps):

        coords = simulation.context.getState(getPositions=True). \
            getPositions(asNumpy=True).in_units_of(angstrom)

        # clears session to avoid running out of memory
        if (i % 1000) == 0:
            tf.keras.backend.clear_session()

        # predict ML forces: predict_on_batch faster for  single structure
        prediction = model.predict_on_batch([np.reshape(coords
            [:ligand_n_atom]/angstrom, (1, -1, 3)),
            np.reshape(ligand_atoms,(1, -1))])
        ML_forces = prediction[0]

        # convert to OpenMM internal units
        ML_forces = np.reshape(ML_forces*kilocalories_per_mole/angstrom, (-1, 3))

        # predict charges
        ligand_charges = predict_charges(prediction, ligand_n_atom, charge_scaling)

        # assign predicted charges to ML atoms
        nbforce = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
        for j in range(ligand_n_atom):
            [old_charge, sigma, epsilon] = nbforce.getParticleParameters(j)
            nbforce.setParticleParameters(j, ligand_charges[j], sigma, epsilon)
            charges[j] = ligand_charges[j]
        nbforce.updateParametersInContext(simulation.context)

        # assign predicted forces to ML atoms
        for j in range(ligand_n_atom):
            ml_force.setParticleParameters(j, j, ML_forces[j])
        ml_force.updateParametersInContext(simulation.context)

        # get total forces
        forces = simulation.context.getState(getForces=True). \
            getForces(asNumpy=True).in_units_of(kilocalories_per_mole / angstrom)

        # check MM contribution to forces (should be 0 for pure ML simulation)
        MM_forces = forces[:ligand_n_atom] - ML_forces

        # advance trajectory one timestep
        simulation.step(1)

        if ((i+1) % print_data) == 0:

            ligand_coords = np.reshape(coords[:ligand_n_atom] / angstrom, (1, -1, 3))

            state = simulation.context.getState(getEnergy=True)
            vels = simulation.context.getState(getVelocities=True).\
                getVelocities(asNumpy=True).value_in_unit(nanometer / picoseconds)
            forces = simulation.context.getState(getForces=True). \
                getForces(asNumpy=True).in_units_of(kilocalories_per_mole / angstrom)

            # predicts energies in kcal/mol
            PE = prediction[1][0][0]

            np.savetxt(f1, ligand_coords[0][:ligand_n_atom])
            np.savetxt(f2, forces[:ligand_n_atom])
            np.savetxt(f3, vels[:ligand_n_atom])
            f4.write(f"{PE}\n")
            np.savetxt(f5, charges[:ligand_n_atom])
            np.savetxt(f6, ML_forces[:ligand_n_atom])
            np.savetxt(f7, MM_forces[:ligand_n_atom])

        if ((i+1) % print_trj) == 0:
            side_length = state.getPeriodicBoxVectors(asNumpy=True)[0,0] / nanometer
            time = simulation.context.getState().getTime().in_units_of(picoseconds)
            vels = simulation.context.getState(getVelocities=True).\
                getVelocities(asNumpy=True).value_in_unit(nanometer / picoseconds)
            vectors = [side_length] * 3
            coords = simulation.context.getState(getPositions=True,
                enforcePeriodicBox=True).getPositions(asNumpy=True).\
                value_in_unit(nanometer)
            grotrj(tot_n_atom, residues, vectors, time,
                coords, vels, gro.atomNames, output_dir, "traj")

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()
    f7.close()

    return None


def predict_charges(prediction, n_atom, charge_scaling):

    # get charge prediction from same network as forces / energies
    ligand_charges = prediction[2].T

    net_charge = sum(ligand_charges[:,0])
    net_charge = round(net_charge)

    # correct predicted partial charges so that ligand has correct net charge
    # then scale them according to user-defined factor
    corr = (sum(ligand_charges) - net_charge) / n_atom
    scale_corr = (charge_scaling - net_charge) / n_atom
    for atm in range(n_atom):
        ligand_charges[atm] = ligand_charges[atm] - corr
        ligand_charges[atm] = charge_scaling * ligand_charges[atm] - scale_corr

    return ligand_charges


def grotrj(n_atom, res, vecs, time, coords, vels, atom_names, output_dir, file_name):
    if time == 0.0:
        open_type = "w"
    else:
        open_type = "a"
    gro_file = open(f"{output_dir}/{file_name}.gro", open_type)
    gro_file.write(f"output t={time}\n")
    gro_file.write(f"{n_atom}\n")
    count = -1
    for i_res in range(len(res)):
        for i_atm in range(len(list(res[i_res].atoms()))):
            count += 1
            gro_file.write(f"{i_res+1:>5}{res[i_res].name:3} "
                f"{atom_names[count]:>6} {count:>4} "
                f"{coords[count][0]:7.3f} {coords[count][1]:7.3f} "
                f"{coords[count][2]:7.3f} {vels[count][0]:7.3f} "
                f"{vels[count][1]:7.3f} {vels[count][2]:7.3f}\n")
    gro_file.write(f"{vecs[0]:10.5f}{vecs[0]:10.5f}{vecs[0]:10.5f}\n")
    gro_file.close()

    return None


class Molecule(object):
    '''
    Base class for coords, forces and energies of an array of molecule
    structures.
    '''

    def get_ZCFE(self, other):
        # For a given set of atom numbers, coords, forces, energies and charges
        # read in, populate the Molecule object which is used to calculate
        # NRFs, decompFE, etc.
        self.atoms = other.atoms
        self.n_atom = other.n_atom
        self.atom_names = other.atom_names
        self.energies = other.energies
        self.elec_energies = other.elec_energies

        if len(other.coords) > 0:
            self.coords = np.reshape(np.vstack(other.coords),
                                     (-1, len(self.atoms), 3))
        if len(other.forces) > 0:
            self.forces = np.reshape(np.vstack(other.forces),
                                     (-1, len(self.atoms), 3))
        if len(other.charges) > 0:
            self.charges = np.reshape(np.vstack(other.charges),
                                     (-1, len(self.atoms)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

