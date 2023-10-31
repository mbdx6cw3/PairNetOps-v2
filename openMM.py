from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmplumed import PlumedForce
from openmmtools import integrators
from sys import stdout
import numpy as np
import output, plumed, read_inputs, os, shutil
from openmmml import MLPotential
from network import Network
import tensorflow as tf
from tensorflow.keras import backend

def setup(pairfenet, ani, plat):

    input_dir = "md_input"
    isExist = os.path.exists(input_dir)
    if not isExist:
        print("Error - no input files detected")
        exit()
    md_params = read_inputs.md(f"{input_dir}/md_params.txt")

    output_dir = "md_output"
    isExist = os.path.exists(output_dir)
    if isExist:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    shutil.copy2(f"./{input_dir}/md_params.txt", f"./{output_dir}/")

    temp = md_params["temp"]
    ts = md_params["ts"]
    bias = md_params["bias"]
    ensemble = md_params["ensemble"]
    thermostat = md_params["thermostat"]
    minim = md_params["minim"]
    coll_freq = md_params["coll_freq"]
    platform = Platform.getPlatformByName(plat)
    gro = GromacsGroFile(f"{input_dir}/input.gro")
    top = GromacsTopFile(f"{input_dir}/input.top",
        periodicBoxVectors=gro.getPeriodicBoxVectors())
    n_atoms = len(gro.getPositions())
    if ani:
        potential = MLPotential('ani2x')
        system = potential.createSystem(top.topology)
    else:
        system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer)

    # for MLP define custom external force and set initial forces to zero
    if pairfenet == True:
        force = CustomExternalForce("-fx*x-fy*y-fz*z")
        system.addForce(force)
        force.addPerParticleParameter("fx")
        force.addPerParticleParameter("fy")
        force.addPerParticleParameter("fz")
        for j in range(n_atoms):
            force.addParticle(j, (0, 0, 0))
    elif pairfenet == False:
        force = 0

    # define ensemble, thermostat and integrator
    if ensemble == "nve":
        integrator = VerletIntegrator(ts*picoseconds)
    elif ensemble == "nvt":
        if thermostat == "nose_hoover":
            integrator = integrators.NoseHooverChainVelocityVerletIntegrator\
                (system, temp*kelvin, coll_freq / picosecond, ts*picoseconds,
                 10, 5, 5)
        elif thermostat == "langevin":
            integrator = LangevinMiddleIntegrator(temp*kelvin,
                coll_freq / picosecond, ts*picoseconds)

    # define biasing potentials
    if bias:
        plumed_file = open(f"{input_dir}/plumed.dat", "r")
        plumed_script = plumed_file.read()
        system.addForce(PlumedForce(plumed_script))

    # set up simulation
    print(platform)
    simulation = Simulation(top.topology, system, integrator, platform)
    simulation.context.setPositions(gro.positions)

    # minimise initial configuration
    if minim:
        simulation.minimizeEnergy()

    return simulation, output_dir, md_params, gro, force

def MD(simulation, pairfenet, output_dir, md_params, gro, force):

    n_steps = md_params["n_steps"]
    print_trj = md_params["print_trj"]
    print_data = md_params["print_data"]
    print_summary = md_params["print_summary"]
    n_atoms = len(gro.getPositions())
    vectors = gro.getUnitCellDimensions().value_in_unit(nanometer)

    if pairfenet == True:
        input_dir = "trained_model"
        isExist = os.path.exists(input_dir)
        if not isExist:
            print("Error - previously trained model could not be located.")
            exit()

        print("Loading a trained model...")
        prescale = np.loadtxt(f"./{input_dir}/prescale.txt",
                          dtype=np.float32).reshape(-1)
        atoms = np.loadtxt(f"./{input_dir}/atoms.txt",
                            dtype=np.float32).reshape(-1)
        ann_params = read_inputs.ann(f"./{input_dir}/ann_params.txt")
        shutil.copy2(f"./{input_dir}/ann_params.txt", f"./{output_dir}")
        mol = read_inputs.Molecule()
        network = Network(mol)
        model = network.build(len(atoms), ann_params, prescale)
        model.summary()
        model.load_weights(f"./{input_dir}/best_ever_model")

    simulation.reporters.append(StateDataReporter(f"./{output_dir}/openmm.csv",
        reportInterval=print_summary,step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, temperature=True, separator=" "))

    f1 = open(f"./{output_dir}/coords.txt", 'w')
    f2 = open(f"./{output_dir}/forces.txt", 'w')
    f3 = open(f"./{output_dir}/velocities.txt", 'w')
    f4 = open(f"./{output_dir}/energies.txt", 'w')

    # loop through total number of timesteps
    for i in range(n_steps):
        coords = simulation.context.getState(getPositions=True). \
            getPositions(asNumpy=True).in_units_of(angstrom)

        if pairfenet == True:
            # clear session every 1000 steps to avoid running out of memory
            if (i % 1000) == 0:
                tf.keras.backend.clear_session()
            prediction = model.predict([np.reshape(coords, (1, -1, 3)),
                                        np.reshape(atoms,(1, -1))])
            forces = prediction[0] * kilocalories_per_mole / angstrom
            forces = np.reshape(forces, (-1, 3))
            for j in range(n_atoms):
                force.setParticleParameters(j, j, forces[j])
            force.updateParametersInContext(simulation.context)

        if (i % print_data) == 0 or i == 0:
            time = simulation.context.getState().getTime()
            velocities = simulation.context.getState(getVelocities=True).\
                getVelocities(asNumpy=True)
            forces = simulation.context.getState(getForces=True).\
                getForces(asNumpy=True)
            state = simulation.context.getState(getEnergy=True)

            if pairfenet == True:
                PE = prediction[2][0][0]
            else:
                PE = state.getPotentialEnergy() / kilojoule_per_mole

            np.savetxt(f1, coords[:n_atoms])
            np.savetxt(f2, forces[:n_atoms])
            np.savetxt(f3, velocities[:n_atoms])
            f4.write(f"{PE}\n")

        if (i % print_trj) == 0 or i == 0:
            output.gro(n_atoms, vectors, time/picoseconds, coords/nanometer,
                       gro.atomNames, output_dir, "output")

        simulation.step(1)

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    return None

