# info...
#!/usr/bin/env python
__author__ = ['Christopher D Williams']
__credits__ = ['CDW', 'Neil Burton', 'Richard Bryce']
__license__ = 'GPL'
__maintainer__ = 'Christopher D Williams'
__email__ = 'christopher.williams@manchester.ac.uk'
__status__ = 'Development'

def main():
    import qm2ml, analyseQM, query_external, openMM, read_inputs, output,\
        analyseMD
    import os, shutil
    import numpy as np
    from network import Network
    from datetime import datetime
    import tensorflow as tf
    from itertools import islice
    import calc_geom

    # read primary user input
    try:
        input_flag = int(input(""" What would you like to do?
            [1] - Run MD simulation.
            [2] - Analyse MD output.
            [3] - Convert MD output into QM input.
            [4] - Analyse QM output.
            [5] - Convert QM output into ML or MD input.
            [6] - Train or Test an ANN.
            [7] - Query external dataset.
            [8] - Generate torsional scan QM input.
            > """))
    except ValueError:
        print("Invalid Value")
        exit()
    except input_flag > 8:
        print("Invalid Value")
        exit()
    print()

    # determine type of calculation to do
    if input_flag == 1:

        startTime = datetime.now()
        option_flag = int(input("""Run MD simulation.
            [1] - Use empirical potential.
            [2] - Use PairFENet potential.
            [3] - Use ANI-2x.
            > """))

        if option_flag == 1:
            print("Use empirical potential.")
            pairfenet = False
            ani = False
        elif option_flag == 2:
            print("Use PairFENet potential.")
            pairfenet = True
            ani = False
        elif option_flag == 3:
            print("Use ANI.")
            pairfenet = False
            ani = True

        plat = str(input("""GPU or CPU?
            > """))
        if plat == "GPU":
            plat = "OpenCL"

        # setup simulation
        simulation, output_dir, md_params, gro, force = \
            openMM.setup(pairfenet, ani, plat)

        # run simulation
        openMM.MD(simulation, pairfenet, ani, output_dir, md_params, gro, force)

        print(datetime.now() - startTime)

    elif input_flag == 2:
        print("Analyse MD output.")

        output_dir = "plots_and_data"
        isExist = os.path.exists(output_dir)
        if not isExist:
            os.makedirs(output_dir)

        option_flag = int(input("""
            [1] - Calculate force S-curve.
            [2] - Calculate force error distribution.
            [3] - Calculate energy correlation.
            [4] - Calculate dihedral angle probability distributions.
            [5] - Calculate 2D free energy surface.
            > """))

        # initiate molecule class for MD dataset
        input_dir1 = "md_output"
        if option_flag == 1 or option_flag == 2 or option_flag == 3 or \
            option_flag == 4:
            while True:
                try:
                    set_size = int(input("Enter the dataset size > "))
                    break
                except ValueError:
                    print("Invalid Value")
            while True:
                try:
                    init = int(input("Enter the initial frame > "))
                    break
                except ValueError:
                    print("Invalid Value")
            mol1 = read_inputs.Molecule()
            read_inputs.dataset(mol1, input_dir1, set_size, "md")

        # initiate molecule class for QM dataset
        if option_flag == 1 or option_flag == 2 or option_flag == 3:
            input_dir2 = "qm_data"
            mol2 = read_inputs.Molecule()
            read_inputs.dataset(mol2, input_dir2, set_size, "qm")

        if option_flag == 1:
            print("Calculating force S-curve...")
            output.scurve(mol2.forces.flatten(), mol1.forces.flatten(),
                output_dir, "mm_f_scurve")
            np.savetxt(f"./{output_dir}/mm_f_test.dat", np.column_stack((
                mol2.forces.flatten(), mol1.forces.flatten())),
                       delimiter=", ", fmt="%.6f")
            # calculate MAE
            mae = 0
            for actual, prediction in zip(mol2.forces.flatten(), mol1.forces.flatten()):
                diff = prediction - actual
                mae += np.sum(abs(diff))
            mae = mae / len(mol2.forces.flatten())
            print(f"Force MAE: {mae}, kcal/mol/A")

        elif option_flag == 2:
            print("Calculating force error distribution...")
            analyseMD.force_MSE_dist(mol2.forces.flatten(),
                mol1.forces.flatten(), output_dir)

        elif option_flag == 3:
            print("Calculating energy correlation with QM...")
            analyseMD.energy_corr(mol2.energies, mol1.energies, output_dir)

        elif option_flag == 4:
            while True:
                try:
                    n_dih = int(input("Enter the number of dihedral angles > "))
                    break
                except ValueError:
                    print("Invalid Value")
                except n_dih > 2:
                    print("Number of dihedral angles can only be 1 or 2")
            CV_list = np.empty(shape=[n_dih, 4], dtype=int)
            for i_dih in range(n_dih):
                atom_indices = input(f"""
                Enter atom indices for dihedral {i_dih+1} separated by spaces:
                e.g. "5 4 6 10"
                Consult mapping.dat for connectivity.
                > """)
                CV_list[i_dih,:] = np.array(atom_indices.split())
            n_bins = int(input("Enter the number of bins > "))
            print("Calculating dihedral angle probability distributions...")
            if n_dih == 1:
                analyseMD.pop1D(mol1, n_bins, CV_list, output_dir, init, set_size)
            elif n_dih == 2:
                analyseMD.pop2D(mol1, n_bins, CV_list, output_dir, init, set_size)

        elif option_flag == 5:
            print("Calculating 2D free energy surface...")
            analyseMD.fes2D(input_dir1, output_dir)

    elif input_flag == 3:
        while True:
            try:
                set_size = int(input("Enter the dataset size > "))
                break
            except ValueError:
                print("Invalid Value")
        while True:
            try:
                init = int(input("Enter the initial frame > "))
                break
            except ValueError:
                print("Invalid Value")
        while True:
            try:
                opt_prop = int(input("Enter % of structures for optimisation > "))
                break
            except ValueError:
                print("Invalid Value")
        output_dir = "qm_input"
        isExist = os.path.exists(output_dir)
        if isExist:
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        input_dir = "md_output"
        isExist = os.path.exists(input_dir)
        if not isExist:
            print("Error - no input files detected")
            exit()

        mol = read_inputs.Molecule()
        read_inputs.dataset(mol, input_dir, set_size, "md")
        output.write_gau(mol, init, set_size, output_dir, opt_prop)

    elif input_flag == 4:
        print("Analyse QM output.")
        while True:
            try:
                set_size = int(input("Enter the dataset size > "))
                break
            except ValueError:
                print("Invalid Value")
        output_dir = "plots_and_data"
        isExist = os.path.exists(output_dir)
        if not isExist:
            os.makedirs(output_dir)

        input_dir = "qm_data"
        isExist = os.path.exists(input_dir)
        if not isExist:
            print("Error - no input files detected")
            exit()

        # initiate molecule class and parse dataset
        mol = read_inputs.Molecule()
        read_inputs.dataset(mol, input_dir, set_size, "qm")

        option_flag = int(input("""
              [1] - Calculate force and energy probability distributions.
              [2] - Calculate interatomic pairwise force components (q).
              [3] - Calculate energy wrt to geometric variable.
              [4] - Calculate distance matrix RMSD.
              [5] - Analyse charges.
              > """))

        if option_flag == 1:
            analyseQM.dist(mol, set_size, output_dir)
        elif option_flag == 2:
            mol.orig_energies = np.copy(mol.energies)
            analyseQM.prescale_e(mol, mol.energies, mol.forces)
            analyseQM.get_eij(mol, set_size, output_dir)
            recomb_F = analyseQM.get_forces(mol, mol.coords, mol.mat_FE)
            np.savetxt(f"./{output_dir}/recomb_test.dat", np.column_stack((
                mol.forces.flatten(), recomb_F.flatten())), delimiter=" ", fmt="%.6f")
        elif option_flag == 3:
            atom_indices = input("""
                Enter atom indices separated by spaces:
                    e.g. for a distance "0 1"
                    e.g. for an angle "1 2 3 4"
                    e.g. for a dihedral "5 4 6 10"
                    Consult mapping.dat for connectivity.
                > """)
            analyseQM.energy_CV(mol, atom_indices, set_size, output_dir)
        elif option_flag == 4:
            print("Calculating distance matrix RMSD...")
            rmsd_dist = analyseQM.rmsd_dist(mol,set_size)
            print(f"Distance matrix RMSD: {np.mean(rmsd_dist)} Angstrom")
        elif option_flag == 5:
            print("Analysing charges")
            #print maximum, minimum and mean charge for each atom
            print(*mol.charges.max(axis=0))
            print(*mol.charges.min(axis=0))
            print(*mol.charges.mean(axis=0))
            exit()

    elif input_flag == 5:

        print("Convert QM output into ML or MD input.")
        option_flag = int(input("""
                     [1] - Convert to ML input.
                     [2] - Convert to MD input (.gro format).
                     > """))
        while True:
            try:
                set_size = int(input("Enter the dataset size > "))
                break
            except ValueError:
                print("Invalid Value")

        while True:
            try:
                step = int(input("Step size > "))
                break
            except ValueError:
                print("Invalid Value")

        if option_flag == 1:
            perm_option = str(input("Shuffle permutations? (Y/N) > "))
            if perm_option == "Y":
                perm = True
            else:
                perm = False
            input_dir = "qm_input"
            isExist = os.path.exists(input_dir)
            if not isExist:
                print("Error - no input files detected")
                exit()
            output_dir = "qm_data"
            isExist = os.path.exists(output_dir)
            if not isExist:
                os.makedirs(output_dir)
            qm2ml.gau2ml(set_size, step, input_dir, output_dir, perm)

        elif option_flag == 2:
            input_dir = "qm_data"
            isExist = os.path.exists(input_dir)
            if not isExist:
                print("Error - no input files detected")
                exit()

            # initiate molecule class and parse dataset
            mol = read_inputs.Molecule()
            read_inputs.dataset(mol, input_dir, set_size, "qm")

            output_dir = "md_input"
            isExist = os.path.exists(output_dir)
            if not isExist:
                os.makedirs(output_dir)

            vectors = [2.5, 2.5, 2.5]
            time = 0.0
            mol.coords = mol.coords / 10 # convert to nm
            for item in range(set_size):
                file_name = str(item+1)
                coord = mol.coords[item][:][:]
                output.gro(mol.n_atom, vectors, time, coord, mol.atom_names,
                    output_dir, file_name)

    elif input_flag == 6:
        startTime = datetime.now()
        option_flag = int(input("""
            [1] - Train a network.
            [2] - Train and test a network.
            [3] - Load and train a network.
            [4] - Load, train and test a network.
            [5] - Load and test a network.
            > """))

        # ensures that tensorflow does not use more cores than requested
        NUMCORES = int(os.getenv("NSLOTS", 1))
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                inter_op_parallelism_threads=NUMCORES,
                allow_soft_placement=True, device_count={'CPU': NUMCORES}))
        tf.compat.v1.keras.backend.set_session(sess)

        # make new directory to store output
        output_dir1 = "plots_and_data"
        isExist = os.path.exists(output_dir1)
        if not isExist:
            os.makedirs(output_dir1)

        # locate dataset
        input_dir2 = "qm_data"
        isExist = os.path.exists(input_dir2)
        if not isExist:
            os.makedirs(input_dir2)

        # initiate molecule and network classes
        mol = read_inputs.Molecule()
        network = Network(mol)
        ann_params = read_inputs.ann("ann_params.txt")
        n_data = ann_params["n_data"]

        # define training and test sets.
        n_train, n_val, n_test = n_data[0], n_data[1], n_data[2]
        set_size = n_train + n_val + n_test
        read_inputs.dataset(mol, input_dir2, set_size, "qm")
        mol.orig_energies = np.copy(mol.energies)

        # set job flags
        if option_flag == 1 or option_flag == 2 or option_flag == 3 or \
                option_flag == 4:
            ann_train = True
            if n_train == 0 or n_val == 0:
                print("""
                ERROR: Cannot train without a training or validation set.
                """)
                exit()
        else:
            ann_train = False
        if option_flag == 2 or option_flag == 4 or option_flag == 5:
            ann_test = True
        else:
            ann_test = False
        if option_flag == 3 or option_flag == 4 or option_flag == 5:
            ann_load = True
        else:
            ann_load = False

        # load previously trained model
        if ann_load:
            input_dir1 = "trained_model"
            isExist = os.path.exists(input_dir1)
            if not isExist:
                print("Error - previously trained model could not be located.")
                exit()
            print("Loading a trained model...")
            prescale = np.loadtxt(f"./{input_dir1}/prescale.txt",
                                  dtype=np.float64).reshape(-1)

            mol.energies = ((prescale[3] - prescale[2]) * (mol.orig_energies
                - prescale[0]) / (prescale[1] - prescale[0]) + prescale[2])
            atoms = np.loadtxt(f"./{input_dir1}/atoms.txt",
                                  dtype=np.float32).reshape(-1)
            model = network.build(len(atoms), ann_params, prescale)
            model.summary()
            model.load_weights(f"./{input_dir1}/best_ever_model")

        else:
            mol.trainval = [*range(0, n_train + n_val, 1)]
            trainval_forces = np.take(mol.forces, mol.trainval, axis=0)
            trainval_energies = np.take(mol.energies, mol.trainval, axis=0)
            trainval_charges = np.take(mol.charges, mol.trainval, axis=0)
            prescale = analyseQM.prescale_e(mol, trainval_energies,
                                            trainval_forces)

        # train model
        if ann_train:

            # open new directory to save newly trained model
            output_dir2 = "trained_model"
            isExist = os.path.exists(output_dir2)
            if not isExist:
                os.makedirs(output_dir2)
            shutil.copy2(f"./ann_params.txt", f"./{output_dir2}")

            # pairwise decomposition of energies
            analyseQM.get_eij(mol, set_size, output_dir1)

            # build model if not training from scratch
            if not ann_load:
                prescale = analyseQM.prescale_eij(mol, prescale)
                print("Building model...")
                model = network.build(len(mol.atoms), ann_params, prescale)
                model.summary()
                np.savetxt(f"./{output_dir2}/atoms.txt",
                           (np.array(mol.atoms)).reshape(-1, 1))
                np.savetxt(f"./{output_dir2}/prescale.txt",
                           (np.array(prescale)).reshape(-1, 1))

            # separate training and validation sets and train network
            print("Training model...")
            mol.train = [*range(0, n_train, 1)]
            mol.val = [*range(n_train, n_train + n_val, 1)]
            network.train(model, mol, ann_params, output_dir1, output_dir2)

            print("Saving model...")
            model.save_weights(f"./{output_dir2}/best_ever_model")

        # test model
        if ann_test:
            mol.test = [*range(n_train + n_val, set_size, 1)]
            if ann_load:
                analyseQM.get_eij(mol, set_size, output_dir1)

            print("Testing model...")
            network.test(model, mol, output_dir1)

        print(datetime.now() - startTime)

    elif input_flag == 7:
        print("Query external dataset.")
        output_dir = "plots_and_data"
        isExist = os.path.exists(output_dir)
        if not isExist:
            os.makedirs(output_dir)
        # options here for MD22/SPICE/etc
        try:
            inp_vsn = input("""
                Enter the dataset version:
                    [1] - original MD17
                    [2] - revised MD17
                > """)
        except ValueError:
            print("Invalid Value")
            exit()
        if int(inp_vsn) == 1:
            source = "md17"
        elif int(inp_vsn) == 2:
            source = "rmd17"
        else:
            print("Invalid Value")
            exit()
        try:
            inp_mol = int(input("""
            Enter the molecule:
                 1 : aspirin
                 2 : azobenzene
                 3 : benzene
                 4 : ethanol
                 5 : malonaldehyde
                 6 : naphthalene
                 7 : paracetamol
                 8 : salicylic
                 9 : toluene
                10 : uracil
            > """))
        except ValueError:
            print("Invalid Value")
            exit()
        if inp_mol > 10 or inp_mol < 1:
            print("Invalid Value")
            exit()
        elif inp_mol == 1:
            molecule = "aspirin"
        elif inp_mol == 2:
            molecule = "azobenzene"
            if inp_vsn == 1:
                print("Invalid value - molecule not in MD17 dataset")
                exit()
        elif inp_mol == 3:
            molecule = "benzene"
        elif inp_mol == 4:
            molecule = "ethanol"
        elif inp_mol == 5:
            molecule = "malonaldehyde"
        elif inp_mol == 6:
            molecule = "naphthalene"
        elif inp_mol == 7:
            molecule = "paracetamol"
            if inp_vsn == 1:
                print("Invalid value - molecule not in MD17 dataset")
                exit()
        elif inp_mol == 8:
            molecule = "salicylic"
        elif inp_mol == 9:
            molecule = "toluene"
        elif inp_mol == 10:
            molecule = "uracil"
        sample_freq = int(input("""
            Sample data every n frames:
            > """))

        while True:
            try:
                n_CV = int(input("Enter the number of CVs > "))
                break
            except ValueError:
                print("Invalid Value")
            except n_CV > 2:
                print("Number of dihedral angles can only be 1 or 2")

        if n_CV == 1:
            query_external.geom(sample_freq, molecule, source, output_dir)
        elif n_CV == 2:
            CV_list = np.empty(shape=[n_CV, 4], dtype=int)
            for i_CV in range(n_CV):
                atom_indices = input(f"""
                Enter atom indices for dihedral {i_CV+1} separated by spaces:
                e.g. "5 4 6 10"
                Consult mapping.dat for connectivity.
                > """)
                CV_list[i_CV,:] = np.array(atom_indices.split())
            n_bins = int(input("Enter the number of bins > "))
            query_external.pop2D(sample_freq, n_bins, CV_list, molecule, source, output_dir)

    # put all this into a module/functions
    elif input_flag == 8:

        element = {1: "H", 6: "C", 7: "N", 8: "O"}
        nuclear_charge_file = open("./nuclear_charges.txt", "r")
        atoms = []
        atom_names = []
        for atom in nuclear_charge_file:
            atoms.append(int(atom))
            atom_names.append(element[atoms[-1]])
        n_atom = len(atoms)

        atom_indices = input(f"""
            Enter atom indices for dihedral separated by spaces:
            e.g. "5 4 6 10"
            Consult mapping.dat for connectivity.
            > """)
        CV_list = [eval(i_atm) for i_atm in atom_indices.split()]

        atom_indices = input(f"""
            Enter atom indices to rotate separated by spaces:
            e.g. "6 10 11"
            Consult mapping.dat for connectivity.
            > """)
        rot_list = [eval(i_rot) for i_rot in atom_indices.split()]

        spacing = int(input("Enter the interval (degrees) > "))
        set_size = int(360/spacing)
        CV = np.empty(shape=[set_size])

        output_dir = "qm_input"
        isExist = os.path.exists(output_dir)
        if not isExist:
            os.makedirs(output_dir)

        # open initial structure, find and extract coordinates
        qm_file = open(f"./mol_1.out", "r")
        for line in qm_file:
            if "Input orientation:" in line:
                coord_block = list(islice(qm_file, 4 + n_atom))[-n_atom:]
        coord = np.empty(shape=[set_size, n_atom, 3])
        for i_atom, atom in enumerate(coord_block):
            coord[:, i_atom] = atom.strip('\n').split()[-3:]
        p = np.zeros([len(CV_list), 3])
        p[0:] = coord[0][CV_list[:]]
        CV[0] = calc_geom.dihedral(p)
        print("Initial torsion angle =", CV[0], "degrees")
        axis = (p[2] - p[1])/ np.linalg.norm(p[2] - p[1])
        # loop through all structures
        for i_angle in range(1, set_size):
            # determine rotation angle for this structure (radians)
            # add option to do reverse scan (times by -1)
            angle = (i_angle * spacing) * np.pi / 180
            # generate rotation matrix for this structure
            mat_rot = generate_rotation_matrix(angle, axis)
            # loop through atoms to be rotated
            for i_atm in range(len(rot_list)):
                # shift to new origin
                old_coord = coord[0][rot_list[i_atm]][:] - coord[0][CV_list[2]][:]
                # rotate old coordinates using rotation matrix
                new_coord = np.matmul(mat_rot,old_coord)
                # new_coord2 = mat_rot.dot(old_coord.T) # equivalent approach
                # shift to old origin
                coord[i_angle][rot_list[i_atm]][:] = new_coord + coord[0][CV_list[2]][:]

        # write gaussian output
        gaussian_opt = open(f"./gaussian_opt.txt", "r")
        text_opt = gaussian_opt.read().strip('\n')
        opt_prop = 1
        CV_list = [i + 1 for i in CV_list]
        for item in range(set_size):
            text = text_opt
            qm_file = open(f"./{output_dir}/mol_{item + 1}.gjf", "w")
            new_text = text.replace("index", f"{item + 1}")
            print(new_text, file=qm_file)
            for atom in range(n_atom):
                print(f"{atom_names[atom]} "
                      f"{coord[item, atom, 0]:.8f} "
                      f"{coord[item, atom, 1]:.8f} "
                      f"{coord[item, atom, 2]:.8f}",
                      file=qm_file)  # convert to Angstroms
            if (item % opt_prop) == 0:
                print(file=qm_file)
                print(*CV_list[:], "B", file=qm_file)
                print(*CV_list[:], "F", file=qm_file)
            print(file=qm_file)
            qm_file.close()
            output.write_pdb(coord[item][:][:], "sali", 1, atoms, atom_names,
                             f"./{output_dir}/mol_{item + 1}.pdb", "w")
        return None


def generate_rotation_matrix(angle, axis):
    #https://en.wikipedia.org/wiki/Rotation_matrix
    #"Rotation matrix from axis and angle"
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    rotation = R.from_rotvec(angle * np.array(axis))
    return rotation.as_matrix()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

