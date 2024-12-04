from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmplumed import PlumedForce
from openmmtools import integrators
import numpy as np
import write_output, read_input, os, shutil
from network import Network
import tensorflow as tf
import random

def setup(force_field):

    input_dir = "md_input"
    isExist = os.path.exists(input_dir)
    if not isExist:
        print("Error - no input files detected")
        exit()
    md_params = read_input.md(f"{input_dir}/md_params.txt")

    output_dir = "md_data"
    isExist = os.path.exists(output_dir)
    if isExist:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    temp = md_params["temp"]
    ts = md_params["ts"]
    ensemble = md_params["ensemble"]
    thermostat = md_params["thermostat"]
    minim = md_params["minim"]
    coll_freq = md_params["coll_freq"]
    gro = GromacsGroFile(f"{input_dir}/input.gro")
    nonbondedmethod = PME
    top = GromacsTopFile(f"{input_dir}/input.top",
                         periodicBoxVectors=gro.getPeriodicBoxVectors())

    # for rigid water to be found the water residue name must be "HOH"
    system = top.createSystem(nonbondedMethod=nonbondedmethod, nonbondedCutoff=1*nanometer,
        constraints=None, removeCMMotion=True, rigidWater=True, switchDistance=None)

    print("Checking simulation setup...")
    print()
    print("Total number of atoms :", top.topology.getNumAtoms())
    print("Total number of residues :", top.topology.getNumResidues())
    print("Total number of bonds :", top.topology.getNumBonds())
    print("Total number of constraints :", system.getNumConstraints())
    print()
    residues = list(top.topology.residues())
    print("Ligand residue name: ", residues[0].name)
    ligand_n_atom = len(list(residues[0].atoms()))
    print("Number of atoms in ligand: ", ligand_n_atom)
    print("Number of bonds in ligand: ", len(list(residues[0].bonds())))

    nb = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
    ewald_tol = nb.getEwaldErrorTolerance()
    [pme_alpha, pme_nx, pme_ny, pme_nz] = nb.getPMEParameters() # TODO: why does this return 0s?
    nb_cut = nb.getCutoffDistance()
    alpha_ewald = (1.0 / nb_cut) * np.sqrt(-np.log(2.0 * ewald_tol))
    print("Periodic boundary conditions? ", system.usesPeriodicBoundaryConditions())
    print("Box dimensions: ", gro.getUnitCellDimensions())
    print("Non-bonded method: ", nb.getNonbondedMethod())
    print("Non-bonded cut-off distance: ", nb_cut)
    print("Ewald sum error tolerance (units?) :", ewald_tol)
    print("PME separation parameter: ", pme_alpha)
    print("Number of PME grid points along each axis: ", pme_nx, pme_ny, pme_nz)
    print("Ewald Gaussian width: ", alpha_ewald)

    if force_field == "empirical":
        ml_force = None

    if force_field == "pair_net":
        # set exceptions for all ligand atoms
        solv_except = nb.getNumExceptions()
        print("Number of solvent exceptions: ", solv_except)
        for i in range(ligand_n_atom):
            for j in range(i):
                nb.addException(i, j, 0, 1, 0)
        print("Number of ligand exceptions: ", nb.getNumExceptions()-solv_except)
        print("Total number of exceptions: ", nb.getNumExceptions())

        # create custom force for PairNet predictions
        ml_force = CustomExternalForce("-fx*x-fy*y-fz*z")
        system.addForce(ml_force)
        ml_force.addPerParticleParameter("fx")
        ml_force.addPerParticleParameter("fy")
        ml_force.addPerParticleParameter("fz")
        for j in range(ligand_n_atom):
            ml_force.addParticle(j, (0, 0, 0))

    # define ensemble, thermostat and integrator
    if ensemble == "nve":
        integrator = VerletIntegrator(ts*picoseconds)
    if ensemble == "npt":
        pressure = md_params["pressure"]
        system.addForce(MonteCarloBarostat(pressure*bar, temp*kelvin))
    if ensemble == "nvt" or ensemble == "npt":
        if thermostat == "nose_hoover":
            integrator = integrators.NoseHooverChainVelocityVerletIntegrator\
                (system, temp*kelvin, coll_freq / picosecond, ts*picoseconds, 10, 5, 5)
            if force_field == "pair_net":
                print("WARNING - are you sure you want to use Nose Hoover with pair-net?")
        elif thermostat == "langevin":
            integrator = LangevinMiddleIntegrator(temp*kelvin,
                coll_freq / picosecond, ts*picoseconds)

    # define biasing potentials
    if md_params["bias"]:
        plumed_file = open(f"{input_dir}/plumed.dat", "r")
        plumed_script = plumed_file.read()
        system.addForce(PlumedForce(plumed_script))
        plumed_file.close()

    # set up simulation
    simulation = Simulation(top.topology, system, integrator)
    simulation.context.setPositions(gro.positions)

    # minimise initial configuration
    if minim:
        simulation.minimizeEnergy()

    # select initial velocities from MB distribution
    if ensemble == "nvt" or ensemble == "npt":
        simulation.context.setVelocitiesToTemperature(temp*kelvin)

    return simulation, system, md_params, gro, top, ml_force, output_dir


def simulate(simulation, system, force_field, md_params, gro, top, ml_force, output_dir):

    max_steps = md_params["max_steps"]
    print_trj = md_params["print_trj"]
    print_data = md_params["print_data"]
    print_summary = md_params["print_summary"]
    tot_n_atom = len(gro.getPositions())

    charges = np.zeros(tot_n_atom)
    nb = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]

    for i in range(system.getNumParticles()):
        charge, sigma, epsilon = nb.getParticleParameters(i)
        charges[i] = charge.value_in_unit(elementary_charge)

    # need to get ligand_n_atom from residue instead
    residues = list(top.topology.residues())
    ligand_n_atom = len(list(residues[0].atoms()))

    if force_field == "pair_net":

        # this is necessary to tell tensorflow to use CPU and GPU when building/predicting
        tf.config.set_visible_devices([], 'GPU')

        mol = read_input.Molecule()
        network = Network(mol)
        print("Loading a trained model...")
        input_dir = "trained_model"
        isExist = os.path.exists(input_dir)
        if not isExist:
            print("ERROR - previously trained model could not be located.")
            exit()
        ligand_atoms = np.loadtxt(f"{input_dir}/nuclear_charges.txt", dtype=np.float32).reshape(-1)
        if len(ligand_atoms) != ligand_n_atom:
            print("ERROR - number of atoms in trained network is incompatible with number of atoms in topology")
            exit()
        mol.n_atom = len(ligand_atoms)
        model = network.load(mol, input_dir)

        # if charges are being predicted enforce ligand net charge
        if md_params["partial_charge"] != "fixed":
            net_charge = md_params["net_charge"]
            charge_model = False

        # load separate network for charge prediction
        if md_params["partial_charge"] == "predicted-sep":
            print("Loading a trained charge model...")
            input_dir = "trained_charge_model"
            isExist = os.path.exists(input_dir)
            if not isExist:
                print("ERROR - previously trained model could not be located.")
                exit()
            charge_model = network.load(mol, input_dir)

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

    if force_field == "pair_net":
        f6 = open(f"./{output_dir}/ML_forces.txt", 'w')
        f7 = open(f"./{output_dir}/MM_forces.txt", 'w')

    if md_params["background_charges"]:
        f8 = open(f"./{output_dir}/background_charges.txt", 'w')

    # run MD simulation for requested number of timesteps
    print("Performing MD simulation...")
    state = simulation.context.getState(getEnergy=True)
    PE = state.getPotentialEnergy() / kilocalories_per_mole
    print("Initial Potential Energy: ", PE, "kcal/mol")

    for i in range(max_steps):

        coords = simulation.context.getState(getPositions=True). \
            getPositions(asNumpy=True).in_units_of(angstrom)

        if force_field == "pair_net":

            # clears session to avoid running out of memory
            if (i % 1000) == 0:
                tf.keras.backend.clear_session()

            # predict ML forces: predict_on_batch faster for  single structure
            prediction = model.predict_on_batch([np.reshape(coords
                [:ligand_n_atom]/angstrom, (1, -1, 3)),
                np.reshape(ligand_atoms,(1, -1))])
            ML_forces = prediction[0]

            # exit simulation if any predicted forces are greater than threshold
            if np.any(ML_forces >= 1000.0):
                print(f"Error - predicted force exceed stability threshold in step {i}")
                print("Final forces:")
                print(ML_forces)
                print()
                print("Final energy:")
                print(prediction[1][0][0])
                print("Ending simulation...")
                break

            # convert to OpenMM internal units
            ML_forces = np.reshape(ML_forces*kilocalories_per_mole/angstrom, (-1, 3))

            # TODO: we surely don't need to do this on every step?
            if md_params["partial_charge"] != "fixed":

                # predict charges
                ligand_charges = predict_charges(md_params, prediction, charge_model,
                    coords, ligand_n_atom, ligand_atoms, net_charge)

                # assign predicted charges to ML atoms
                nbforce = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
                for j in range(ligand_n_atom):
                    [old_charge, sigma, epsilon] = nbforce.getParticleParameters(j)
                    nbforce.setParticleParameters(j, ligand_charges[j], sigma, epsilon)
                    charges[j] = ligand_charges[j]  # TODO: units???
                nbforce.updateParametersInContext(simulation.context)

            # assign predicted forces to ML atoms
            for j in range(ligand_n_atom):
                ml_force.setParticleParameters(j, j, ML_forces[j])
            ml_force.updateParametersInContext(simulation.context)

        # get total forces
        forces = simulation.context.getState(getForces=True). \
            getForces(asNumpy=True).in_units_of(kilocalories_per_mole / angstrom)

        # check MM contribution to forces (should be 0 for pure ML simulation)
        if force_field == "pair_net":
            MM_forces = forces[:ligand_n_atom] - ML_forces

        # advance trajectory one timestep
        simulation.step(1)

        if (i % print_data) == 0:

            # if using background charges we need to wrap coordinates and shift so that amine is at centre
            if md_params["background_charges"]:
                # get box length
                side_length = state.getPeriodicBoxVectors(asNumpy=True)[0, 0] / angstrom
                coords = simulation.context.getState(getPositions=True,
                    enforcePeriodicBox=False).getPositions(asNumpy=True). \
                    value_in_unit(angstrom)
                # shift ligand box so that ligand is in the centre
                coords[:][:] = coords[:][:] - coords[0][:] + (side_length / 2)
                # wrap coordinates
                coords = coords - np.floor(coords / side_length) * side_length
                # write background charge coordinates
                background_charges = np.hstack((charges[:, np.newaxis], coords))
                ligand_coords = np.reshape(coords[:ligand_n_atom], (1, -1, 3))
            else:
                ligand_coords = np.reshape(coords[:ligand_n_atom] / angstrom, (1, -1, 3))

            state = simulation.context.getState(getEnergy=True)
            vels = simulation.context.getState(getVelocities=True).\
                getVelocities(asNumpy=True).value_in_unit(nanometer / picoseconds)
            forces = simulation.context.getState(getForces=True). \
                getForces(asNumpy=True).in_units_of(kilocalories_per_mole / angstrom)

            # predicts energies in kcal/mol
            if force_field == "pair_net":
                PE = prediction[1][0][0]
            else:
                PE = state.getPotentialEnergy() / kilocalories_per_mole

            np.savetxt(f1, ligand_coords[0][:ligand_n_atom])
            np.savetxt(f2, forces[:ligand_n_atom])
            np.savetxt(f3, vels[:ligand_n_atom])
            f4.write(f"{PE}\n")
            np.savetxt(f5, charges[:ligand_n_atom])
            if force_field == "pair_net":
                np.savetxt(f6, ML_forces[:ligand_n_atom])
                np.savetxt(f7, MM_forces[:ligand_n_atom])
            if md_params["background_charges"]:
                np.savetxt(f8, background_charges[ligand_n_atom:])

        if (i % print_trj) == 0:
            side_length = state.getPeriodicBoxVectors(asNumpy=True)[0,0] / nanometer
            time = simulation.context.getState().getTime().in_units_of(picoseconds)
            vels = simulation.context.getState(getVelocities=True).\
                getVelocities(asNumpy=True).value_in_unit(nanometer / picoseconds)
            vectors = [side_length] * 3
            coords = simulation.context.getState(getPositions=True,
                enforcePeriodicBox=True).getPositions(asNumpy=True).\
                value_in_unit(nanometer)
            write_output.grotrj(tot_n_atom, residues, vectors, time,
                coords, vels, gro.atomNames, output_dir, "traj")

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    if force_field == "pair_net":
        f6.close()
        f7.close()

    # print final state to .gro
    print("Printing final state...")
    print_final(state, simulation, tot_n_atom, residues, gro, output_dir)
    return None


def predict_charges(md_params, prediction, charge_model, coords, n_atom,
                    ligand_atoms, net_charge):

    # get charge prediction from same network as forces / energies
    if md_params["partial_charge"] == "predicted":
        ligand_charges = prediction[2].T

    # get charge prediction from separate network
    elif md_params["partial_charge"] == "predicted-sep":
        charge_prediction = charge_model.predict_on_batch(
            [np.reshape(coords[:n_atom] / angstrom, (1, -1, 3)),
             np.reshape(ligand_atoms, (1, -1))])
        ligand_charges = charge_prediction[2].T

    # correct predicted partial charges so that ligand has correct net charge
    corr = (sum(ligand_charges) - net_charge) / n_atom
    for atm in range(n_atom):
        ligand_charges[atm] = ligand_charges[atm] - corr

    return ligand_charges


def permute(ligand_coords, n_perm_grp, perm_atm, n_symm, n_symm_atm):
    # loop through symmetry groups
    for i_perm in range(n_perm_grp):
        # perform 10 swap moves for this symmetry group
        for i_swap in range(10):
            # for this permutation randomly select a symmetry group
            old_perm = perm_atm[i_perm][random.randint(0,n_symm[i_perm]-1)][:]
            new_perm = perm_atm[i_perm][random.randint(0,n_symm[i_perm]-1)][:]
            # swap and save coordinates for these groups
            for i_atm in range(n_symm_atm[i_perm]):
                temp_coord = np.copy(ligand_coords[0, old_perm[i_atm] - 1])
                ligand_coords[0, old_perm[i_atm] - 1] = \
                    ligand_coords[0, new_perm[i_atm] - 1]
                ligand_coords[0, new_perm[i_atm] - 1] = temp_coord

    return ligand_coords


def print_final(state, simulation, tot_n_atom, residues, gro, output_dir):
    side_length = state.getPeriodicBoxVectors(asNumpy=True)[0, 0] / nanometer
    time = simulation.context.getState().getTime().in_units_of(picoseconds)
    vels = simulation.context.getState(getVelocities=True). \
        getVelocities(asNumpy=True).value_in_unit(nanometer / picoseconds)
    coords = simulation.context.getState(getPositions=True,
                                         enforcePeriodicBox=True).getPositions(
        asNumpy=True).value_in_unit(nanometer)
    vectors = [side_length] * 3
    write_output.grotrj(tot_n_atom, residues, vectors, time,
                        coords, vels, gro.atomNames, output_dir, "final")
    return