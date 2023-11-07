from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmplumed import PlumedForce
from openmmtools import integrators
import numpy as np
import output, read_inputs, os, shutil
from openmmml import MLPotential
from network import Network
import tensorflow as tf
from datetime import datetime

#TODO: can unused modules be removed?

def setup(pairfenet, ani2x, empirical, plat):

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

    # TODO: set up of ML/MM system. mixed system: https://github.com/openmm/openmm-ml
    if ani2x:
        potential = MLPotential('ani2x')
        system = potential.createSystem(top.topology)
    else:
        system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer)

    # define non-bonded force which contains empirical potential parameters
    #nb_force = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
    #for i in range(system.getNumParticles()):
    #    print(nb_force.getParticleParameters(i))

    # for MLP define custom external force and set initial forces to zero
    if pairfenet == True:
        force = CustomExternalForce("-fx*x-fy*y-fz*z")
        system.addForce(force)
        force.addPerParticleParameter("fx")
        force.addPerParticleParameter("fy")
        force.addPerParticleParameter("fz")
        for j in range(n_atoms):
            force.addParticle(j, (0, 0, 0))
    elif empirical == True or ani2x == True:
        force = 0 # no custom force, i.e. just use potential as defined in .top

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

    return simulation, output_dir, md_params, gro, force

def MD(simulation, pairfenet, ani2x, empirical, output_dir, md_params, gro, force):

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

    # this prevents tensorflow printing warnings or other information
    tf.get_logger().setLevel('ERROR')

    #init_coords = simulation.context.getState(
        #getPositions=True).getPositions(asNumpy=True).in_units_of(
        #angstrom)

    f1 = open(f"./{output_dir}/coords.txt", 'w')
    f2 = open(f"./{output_dir}/forces.txt", 'w')
    f3 = open(f"./{output_dir}/velocities.txt", 'w')
    f4 = open(f"./{output_dir}/energies.txt", 'w')
    f5 = open(f"./{output_dir}/charges.txt", 'w')

    # run MD simulation for requested number of timesteps
    for i in range(n_steps):

        coords = simulation.context.getState(getPositions=True). \
            getPositions(asNumpy=True).in_units_of(angstrom)

        #coords = translate_coords(init_coords / angstrom,
                                            #coords / angstrom, atoms)
        #simulation.context.setPositions(coords * angstrom)

        if ani2x == True:
            charges = np.zeros(n_atoms)

        if pairfenet == True:

            # clear session to avoid running out of memory
            if (i % 1000) == 0:
                tf.keras.backend.clear_session()

            # predict forces and convert to internal OpenMM units
            startTime = datetime.now()
            print(np.reshape(coords, (1, -1, 3)),np.reshape(atoms,(1, -1)))
            prediction = model.predict([np.reshape(coords, (1, -1, 3)),
                                        np.reshape(atoms,(1, -1))], verbose=0)
            print(datetime.now()-startTime)
            forces = np.reshape(prediction[0]*kilocalories_per_mole/angstrom, (-1, 3))

            # assign forces to ML atoms
            for j in range(n_atoms):
                force.setParticleParameters(j, j, forces[j])
            force.updateParametersInContext(simulation.context)

            # TODO: dynamically reassign charges to ML atoms
            # set charges to zero for now
            charges = np.zeros(n_atoms)

        if (i % print_data) == 0 or i == 0:
            time = simulation.context.getState().getTime()
            state = simulation.context.getState(getEnergy=True)
            velocities = simulation.context.getState(getVelocities=True).\
                getVelocities(asNumpy=True)
            forces = simulation.context.getState(getForces=True). \
                getForces(asNumpy=True).in_units_of(kilocalories_per_mole / angstrom)

            # if not using pairfenet convert forces to kcal/mol/A before printing
            if pairfenet == False:
                forces = simulation.context.getState(getForces=True).\
                    getForces(asNumpy=True).in_units_of(kilocalories_per_mole/angstrom)
                # TODO: reassign charges using prediction from trained network
                # charges = nb_force.getParticleParameters(0)???

            # predicts energies in kcal/mol
            if pairfenet == True:
                PE = prediction[1][0][0]
            else:
                PE = state.getPotentialEnergy() / kilocalories_per_mole

            np.savetxt(f1, coords[:n_atoms])
            np.savetxt(f2, forces[:n_atoms])
            np.savetxt(f3, velocities[:n_atoms])
            f4.write(f"{PE}\n")
            np.savetxt(f5, charges[:n_atoms])

        if (i % print_trj) == 0 or i == 0:
            output.gro(n_atoms, vectors, time/picoseconds, coords/nanometer,
                       gro.atomNames, output_dir, "output")

        simulation.step(1)

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    return None



def translate_coords(init_coords, coords, atoms):
    '''translate molecule center-of-mass to centre of box'''
    _ZM = {1: 1.008, 6: 12.011, 8: 15.999, 7: 14.0067}
    n_atoms = len(atoms)
    masses = np.array([_ZM[a] for a in atoms])
    _PA, _MI, com = get_principal_axes(coords, masses)
    c_translated = np.zeros((n_atoms,3))
    c_rotated = np.zeros((n_atoms,3))
    for i in range(n_atoms):
        c_translated[i] = coords[i] - com
        for j in range(3):
            c_rotated[i][j] = np.dot(c_translated[i], _PA[j])
    return c_translated

def get_principal_axes(coords, masses):
    com  = get_com(coords, masses)
    moi = get_moi_tensor(com, coords, masses)
    eigenvalues, eigenvectors = np.linalg.eig(moi) #diagonalise moi
    transposed = np.transpose(eigenvectors) #turn columns to rows

    ##find min and max eigenvals (magnitudes of principal axes/vectors)
    min_eigenvalue = abs(eigenvalues[0])
    if eigenvalues[1] < min_eigenvalue:
        min_eigenvalue = eigenvalues[1]
    if eigenvalues[2] < min_eigenvalue:
        min_eigenvalue = eigenvalues[2]

    max_eigenvalue = abs(eigenvalues[0])
    if eigenvalues[1] > max_eigenvalue:
        max_eigenvalue = eigenvalues[1]
    if eigenvalues[2] > max_eigenvalue:
        max_eigenvalue = eigenvalues[2]

    #PA = principal axes
    _PA = np.zeros((3,3))
    #MI = moment of inertia
    _MI = np.zeros((3))
    for i in range(3):
        if eigenvalues[i] == max_eigenvalue:
            _PA[i] = transposed[i]
            _MI[i] = eigenvalues[i]
        elif eigenvalues[i] == min_eigenvalue:
            _PA[i] = transposed[i]
            _MI[i] = eigenvalues[i]
        else:
            _PA[i] = transposed[i]
            _MI[i] = eigenvalues[i]

    return _PA, _MI, com


def get_com(coords, masses):
    com = np.zeros((3))
    for coord, mass in zip(coords, masses):
        for i in range(3):
            com[i] += mass * coord[i]
    com = com / sum(masses)
    return np.array(com)


def get_moi_tensor(com, coords, masses):
    x_cm, y_cm, z_cm = com[0], com[1], com[2]
    _I = np.zeros((3,3))
    for coord, mass in zip(coords, masses):
        _I[0][0] += (abs(coord[1] - y_cm)**2 + \
                abs(coord[2] - z_cm)**2) * mass
        _I[0][1] -= (coord[0] - x_cm) * (coord[1] - y_cm) * mass
        _I[1][0] -= (coord[0] - x_cm) * (coord[1] - y_cm) * mass

        _I[1][1] += (abs(coord[0] - x_cm)**2 + \
                abs(coord[2] - z_cm)**2) * mass
        _I[0][2] -= (coord[0] - x_cm) * (coord[2] - z_cm) * mass
        _I[2][0] -= (coord[0] - x_cm) * (coord[2] - z_cm) * mass

        _I[2][2] += (abs(coord[0] - x_cm)**2 + \
                abs(coord[1] - y_cm)**2) * mass
        _I[1][2] -= (coord[1] - y_cm) * (coord[2] - z_cm) * mass
        _I[2][1] -= (coord[1] - y_cm) * (coord[2] - z_cm) * mass
    return _I
