from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmplumed import PlumedForce
from openmmtools import integrators
import numpy as np
import write_output, read_input, os, shutil
from openmmml import MLPotential
from network import Network
import tensorflow as tf

def setup(force_field, plat):

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

    # TODO: set up of ML/MM system. mixed system: https://github.com/openmm/openmm-ml
    if force_field == "ani":
        potential = MLPotential('ani2x')
        system = potential.createSystem(top.topology)
    else:
        # for rigid water to be found the water residue name must be "HOH"
        system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer)
            #ewaldErrorTolerance=0.0005, constraints=None, removeCMMotion=True,
            #rigidWater=True, switchDistance=None)

    #print(system.getNumConstraints())

    # define non-bonded force which contains empirical potential parameters
    #nb_force = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
    #for i in range(system.getNumParticles())
    #    print(nb_force.getParticleParameters(i))
    #exit()

    # for MLP define custom external force and set initial forces to zero
    force = 0
    if force_field == "pair_net":
        force = CustomExternalForce("-fx*x-fy*y-fz*z")
        system.addForce(force)
        force.addPerParticleParameter("fx")
        force.addPerParticleParameter("fy")
        force.addPerParticleParameter("fz")
        for j in range(n_atoms):
            force.addParticle(j, (0, 0, 0))

    # define ensemble, thermostat and integrator
    if ensemble == "nve":
        integrator = VerletIntegrator(ts*picoseconds)
    elif ensemble == "nvt":
        if thermostat == "nose_hoover":
            integrator = integrators.NoseHooverChainVelocityVerletIntegrator\
                (system, temp*kelvin, coll_freq / picosecond, ts*picoseconds, 10, 5, 5)
            if force_field == "pair_net":
                print("WARNING - are you sure you want to use this combination of thermostat and potential?")
        elif thermostat == "langevin":
            integrator = LangevinMiddleIntegrator(temp*kelvin,
                coll_freq / picosecond, ts*picoseconds)
            # TODO: what is the difference between picoseconds and picosecond?

    # define biasing potentials
    # TODO: there is a problem here - it seems ANI is incompatible with bias. Why?
    if bias:
        plumed_file = open(f"{input_dir}/plumed.dat", "r")
        plumed_script = plumed_file.read()
        system.addForce(PlumedForce(plumed_script))

    # set up simulation
    simulation = Simulation(top.topology, system, integrator, platform)
    simulation.context.setPositions(gro.positions)

    # minimise initial configuration
    if minim:
        simulation.minimizeEnergy()
    if ensemble == "nvt":
        simulation.context.setVelocitiesToTemperature(temp*kelvin)

    # check everything is set up correctly and print to output
    #nb = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
    #for i in range(system.getNumParticles()):
    #    charge, sigma, epsilon = nb.getParticleParameters(i)
     #   print(charge,sigma,epsilon)
    #print(nb.getEwaldErrorTolerance())
    #print(nb.getNonbondedMethod())
    #[alpha_ewald, nx, ny, nz] = nb.getPMEParameters()
    #print(nx,ny,nz)
    #alpha_ewald = (1.0 / nb.getCutoffDistance()) * np.sqrt(-np.log(2.0 * nb.getEwaldErrorTolerance()))
    #print(alpha_ewald)
    #print(nb.getCutoffDistance())
    #print(gro.getPeriodicBoxVectors())

    # TODO: somewhere we need to define whether charges will come from a separate ANN or the same one as energy and forces

    return simulation, system, output_dir, md_params, gro, force

def simulate(simulation, system, force_field, output_dir, md_params, gro, force):

    n_steps = md_params["n_steps"]
    print_trj = md_params["print_trj"]
    print_data = md_params["print_data"]
    print_summary = md_params["print_summary"]
    n_atoms = len(gro.getPositions())
    vectors = gro.getUnitCellDimensions().value_in_unit(nanometer)

    if force_field == "pair_net":
        print("Loading a trained model...")
        mol = read_input.Molecule()
        network = Network(mol)
        ann_params = read_input.ann("trained_model/ann_params.txt")
        atoms = np.loadtxt("trained_model/nuclear_charges.txt", dtype=np.float32).reshape(-1)
        mol.n_atom = n_atoms
        model = network.load(mol, ann_params)

    if force_field == "empirical":
        nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
        charges = np.zeros(n_atoms)
        for i in range(system.getNumParticles()):
            charge, sigma, epsilon = nonbonded.getParticleParameters(i)
            charges[i] = charge.value_in_unit(elementary_charge)

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

    # run MD simulation for requested number of timesteps
    for i in range(n_steps):

        coords = simulation.context.getState(getPositions=True). \
            getPositions(asNumpy=True).in_units_of(angstrom)

        ##velocities = simulation.context.getState(getVelocities=True). \
        #    getVelocities(asNumpy=True)

        #print(velocities)
        #exit()

        if i == 0:
            write_output.gro(n_atoms, vectors, "0.0000", coords / nanometer,
                gro.atomNames, output_dir, "output")

        if force_field == "ani":
            charges = np.zeros(n_atoms)

        if force_field == "pair_net":
            if (i % 1000) == 0: # clears session to avoid running out of memory
                tf.keras.backend.clear_session()

            # predict forces - predict_on_batch faster with only single structure
            prediction = model.predict_on_batch([np.reshape(coords
                [:n_atoms]/angstrom, (1, -1, 3)), np.reshape(atoms,(1, -1))])
            # convert to OpenMM internal units
            forces = np.reshape(prediction[0]*kilocalories_per_mole/angstrom, (-1, 3))

            # assign forces to ML atoms
            for j in range(n_atoms):
                force.setParticleParameters(j, j, forces[j])
            force.updateParametersInContext(simulation.context)

            # TODO: dynamically reassign charges to ML atoms
            # set charges to zero for now
            charges = prediction[2].T

        # advance trajectory one timestep
        simulation.step(1)

        # print output
        if (i % print_data) == 0 or i == 0:
            time = simulation.context.getState().getTime()
            state = simulation.context.getState(getEnergy=True)
            coords = simulation.context.getState(getPositions=True). \
                getPositions(asNumpy=True).in_units_of(angstrom)
            velocities = simulation.context.getState(getVelocities=True).\
                getVelocities(asNumpy=True)
            forces = simulation.context.getState(getForces=True). \
                getForces(asNumpy=True).in_units_of(kilocalories_per_mole / angstrom)

            # if not using pairfenet convert forces to kcal/mol/A before printing
            if force_field == "pairnet":
                forces = simulation.context.getState(getForces=True).\
                    getForces(asNumpy=True).in_units_of(kilocalories_per_mole/angstrom)
                # TODO: reassign charges using prediction from trained network
                # charges = nb_force.getParticleParameters(0)???

            # predicts energies in kcal/mol
            if force_field == "pairnet":
                PE = prediction[1][0][0]
            else:
                PE = state.getPotentialEnergy() / kilocalories_per_mole

            np.savetxt(f1, coords[:n_atoms])
            np.savetxt(f2, forces[:n_atoms])
            np.savetxt(f3, velocities[:n_atoms])
            f4.write(f"{PE}\n")
            np.savetxt(f5, charges[:n_atoms])

        #TODO: providing PBCs are actually applied need to wrap coords here
        #TODO: do we need to do this for coords above too?
        if (i % print_trj) == 0:
            write_output.gro(n_atoms, vectors, time/picoseconds, coords/nanometer,
                       gro.atomNames, output_dir, "output")

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    return None

