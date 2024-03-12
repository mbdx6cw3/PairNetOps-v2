# info...
#!/usr/bin/env python
__author__ = ['Christopher D Williams']
__credits__ = ['CDW', 'Neil Burton', 'Richard Bryce']
__license__ = 'GPL'
__maintainer__ = 'Christopher D Williams'
__email__ = 'christopher.williams@manchester.ac.uk'
__status__ = 'Development'

def main():
    import analysis, md, read_input, write_output
    import os, shutil
    import numpy as np
    from network import Network
    from datetime import datetime

    # read primary user input
    try:
        input_flag = int(input("""What would you like to do?
            [1] - Run a Molecular Dynamics Simulation.
            [2] - Train or Test a PairNet Potential.
            [3] - Analyse an Existing Dataset.
            [4] - Generate a New Dataset.
            [5] - Reformat an Existing Dataset.
            > """))

        if input_flag > 5 or input_flag < 1:
            exit("Invalid Value")
    except ValueError:
        exit("Invalid Value")
    print()


    # determine type of calculation to do
    if input_flag == 1:

        option_flag = int(input("""Run a Molecular Dynamics Simulation.
            [1] - Use an Empirical Potential.
            [2] - Use PairNet
            [3] - Use ANI-2x.
            [4] - Use MACE-OFF
            > """))

        if option_flag == 1:
            print("Use an Empirical Potential.")
            potential = "empirical"
        elif option_flag == 2:
            print("Use PairNet.")
            potential = "pair_net"
        elif option_flag == 3:
            print("Use ANI-2x.")
            potential = "ani"
        elif option_flag == 4:
            print("Use MACE-OFF")
            potential = "mace-off"
        print()

        plat = str(input("""GPU or CPU?
            > """))
        if plat == "GPU":
            plat = "OpenCL"
            print("Using a GPU node")
        else:
            print("Using a CPU node")
        print()
        print("Running MD Simulation...")

        # setup simulation
        simulation, system, output_dir, md_params, gro, top, ml_force = md.setup(potential, plat)

        # run simulation
        startTime = datetime.now()
        md.simulate(simulation, system, potential, output_dir, md_params, gro, top, ml_force)
        print(datetime.now() - startTime)

    elif input_flag == 2:

        option_flag = int(input("""Train or Test a PairNet Potential.
             [1] - Train a network.
             [2] - Train and test a network.
             [3] - Load and train a network.
             [4] - Load, train and test a network.
             [5] - Load and test a network.
             > """))

        # make new directory to store output
        output_dir1 = "plots_and_data"
        isExist = os.path.exists(output_dir1)
        if not isExist:
            os.makedirs(output_dir1)

        # initiate molecule and network classes
        mol = read_input.Molecule()
        network = Network(mol)
        ann_params = read_input.ann("ann_params.txt")

        # define training and test sets.
        n_data = ann_params["n_data"]
        n_train, n_val, n_test = n_data[0], n_data[1], n_data[2]
        size = n_train + n_val + n_test

        # locate and read in dataset
        input_dir = "ml_data"
        isExist = os.path.exists(input_dir)
        if not isExist:
            os.makedirs(input_dir)
        read_input.Dataset(mol, size, 0, 1, input_dir, "txt")
        mol.orig_energies = np.copy(mol.energies)

        # set job flags
        if option_flag == 1 or option_flag == 2 or option_flag == 3 or \
                option_flag == 4:
            ann_train = True
            if n_train == 0 or n_val == 0:
                print("ERROR - can't train without a dataset.")
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

        conf_test = False
        if option_flag == 5:
            test_option = str(input("Test conformational distribution? (Y/N) > "))
            if test_option == "Y":
                conf_test = True
            else:
                conf_test = False

        # load previously trained model
        if ann_load:
            print("Loading a trained model...")
            input_dir = "trained_model"
            isExist = os.path.exists(input_dir)
            if not isExist:
                print("ERROR - previously trained model could not be located.")
                exit()
            model = network.load(mol, input_dir)

        else:
            mol.trainval = [*range(0, n_train + n_val, 1)]
            trainval_forces = np.take(mol.forces, mol.trainval, axis=0)
            trainval_energies = np.take(mol.energies, mol.trainval, axis=0)
            norm_scheme = ann_params["norm_scheme"]
            prescale = analysis.prescale_e(mol, trainval_energies,
                trainval_forces, norm_scheme)

        # train model
        if ann_train:

            # open new directory to save newly trained model
            output_dir2 = "trained_model"
            isExist = os.path.exists(output_dir2)
            if not isExist:
                os.makedirs(output_dir2)
            shutil.copy2(f"./ann_params.txt", f"./{output_dir2}")
            np.savetxt(f"./{output_dir2}/nuclear_charges.txt",
                (np.array(mol.atoms)).reshape(-1, 1))

            # pairwise decomposition of energies
            print("Calculating pairwise energies...")
            analysis.get_eij(mol, size, output_dir1)

            # build model if not training from scratch
            if not ann_load:
                # get prescaling factors
                prescale = analysis.prescale_eij(mol, prescale)
                np.savetxt(f"./{output_dir2}/prescale.txt",
                           (np.array(prescale)).reshape(-1, 1))
                print("Building model...")
                model = network.build(mol, ann_params, prescale)

            # separate training and validation sets and train network
            mol.train = [*range(0, n_train, 1)]
            mol.val = [*range(n_train, n_train + n_val, 1)]
            print("Training model...")
            startTime = datetime.now()
            network.train(model, mol, ann_params, output_dir1, output_dir2)
            print(datetime.now() - startTime)

            print("Saving model...")
            model.save_weights(f"./{output_dir2}/best_ever_model")

        # test model
        if ann_test:
            mol.test = [*range(n_train + n_val, size, 1)]
            if ann_load:
                print("Calculating pairwise energies...")
                analysis.get_eij(mol, size, output_dir1)

            print("Testing model...")
            network.test(model, mol, output_dir1, ann_params, conf_test)

    elif input_flag == 3:
        print("Analyse a Dataset")

        output_dir = "plots_and_data"
        isExist = os.path.exists(output_dir)
        if not isExist:
            os.makedirs(output_dir)

        option_flag = int(input("""What would you like to analyse?
        [1] - Analyse Forces and Energies.
        [2] - Assess Stability.
        [3] - Analyse Geometry.
        [4] - Analyse Charges.
        [5] - Compare Datasets.
        [6] - Calculate Multidimensional Free Energy Surface.
        > """ ))
        print()

        if option_flag < 6:
            while True:
                try:
                    size = int(input("Enter number of structures > "))
                    init = 0
                    #init = int(input("Enter the initial structure > "))
                    #space = int(input("Enter spacing between structures > "))
                    space = 1
                    print()
                    break
                except ValueError:
                    exit("Invalid Value")

        if option_flag < 5:
            try:
                input_type = int(input("""Type of Dataset to Analyse:
                [1] - ml_data (.txt)
                [2] - md_data (.txt)
                [3] - qm_data (.out)
                [4] - External.
                > """))
                if input_type > 3 or input_type < 1:
                    exit("Invalid Value")
            except ValueError:
                exit("Invalid Value")
            print()

            if input_type == 1:
                input_dir = "ml_data"
                isExist = os.path.exists(input_dir)
                if not isExist:
                    print("Error - no input files in the working directory")
                    exit()
                mol = read_input.Molecule()
                read_input.Dataset(mol, size, init, space, input_dir, "txt")
                # TODO: size needs to change after this

            elif input_type == 2:
                input_dir = "md_data"
                isExist = os.path.exists(input_dir)
                if not isExist:
                    print("Error - no input files in the working directory")
                    exit()
                mol = read_input.Molecule()
                read_input.Dataset(mol, size, init, space, input_dir, "txt")

            elif input_type == 3:
                input_dir = "qm_data"
                isExist = os.path.exists(input_dir)
                if not isExist:
                    print("Error - no input files in the working directory")
                    exit()
                mol = read_input.Molecule()
                read_input.Dataset(mol, size, init, space, input_dir, "gau")

            elif input_type == 4:
                input_dir = "/Users/user/datasets"
                mol = read_input.Molecule()
                read_input.Dataset(mol, size, init, space, input_dir, "ext")

        elif option_flag == 5:

            input_dir1 = "md_data"
            isExist = os.path.exists(input_dir1)
            if not isExist:
                print("Error - no input files in the working directory.")
                exit()
            mol1 = read_input.Molecule()
            read_input.Dataset(mol1, size, init, space, input_dir1, "txt")

            input_dir2 = "ml_data"
            isExist = os.path.exists(input_dir2)
            if not isExist:
                print("Error - no input files in the working directory.")
                exit()
            mol2 = read_input.Molecule()
            read_input.Dataset(mol2, size, init, space, input_dir2, "txt")

        if option_flag == 1:
            print("Analyse Forces and Energies.")

            print("Calculating force and energy distributions...")
            analysis.dist(mol, size, output_dir)

            print("Calculating pairwise energy components, e_ij...")
            mol.orig_energies = np.copy(mol.energies)
            norm_scheme = "force"
            analysis.prescale_e(mol, mol.energies, mol.forces, norm_scheme)
            analysis.get_eij(mol, size, output_dir)

        elif option_flag == 2:
            print("Analyse Stability.")
            analysis.check_stability(mol, size, output_dir)

        elif option_flag == 3:
            print("Analyse Geometry.")

            geom_flag = int(input("""Analysis to Perform:
            [1] - Get energy vs geometric variable.
            [2] - Get root mean squared deviation of distance matrix.
            [3] - Get 1D probability distribution of geometric variable.
            [4] - Get 2D probability distribution of geometric variable.
            > """))

            if geom_flag == 1:
                print("Get energy vs geometric variable.")
                n_bins = int(input("Enter the number of bins > "))
                CV_list = analysis.getCVs(1)
                analysis.energy_CV(mol, n_bins, CV_list[0], size, output_dir)

            elif geom_flag == 2:
                print("Get root mean squared deviation of distance matrix.")
                rmsd_dist = analysis.rmsd_dist(mol, size)
                print(f"Distance matrix RMSD: {np.mean(rmsd_dist)} Angstrom")

            elif geom_flag == 3:
                print("Get 1D probability distribution of geometric variable.")
                n_bins = int(input("Enter the number of bins > "))
                CV_list = analysis.getCVs(1)
                analysis.pop1D(mol, n_bins, CV_list[0], output_dir, size)

            elif geom_flag == 4:
                print("Get 2D probability distribution of geometric variable.")
                n_bins = int(input("Enter the number of bins > "))
                CV_list = analysis.getCVs(2)
                analysis.pop2D(mol, n_bins, CV_list, output_dir, size)

        elif option_flag == 4:

            print("Analyse Charges.")
            charge_option = int(input("""
                [1] Calculate mean partial charges.
                [2] Calculate partial charge probability distribution.
                [3] Calculate partial charge vs geometric variable.
                [4] Calculate intramolecular electrostatic potential energy.
                > """))
            print()

            if charge_option == 1:
                print("Calculating mean partial charges...")
                np.savetxt(f"./{output_dir}/mean_charges.dat", np.column_stack((
                    np.arange(mol.n_atom), mol.charges.mean(axis=0))),
                           fmt="%d %.6f", delimiter=" ")

            elif charge_option == 2:
                atom = int(input("""Enter atom index > """))
                print("Calculating partial charge probability distribution...")
                analysis.charge_dist(mol, atom, size, output_dir)

            elif charge_option == 3:
                atom = int(input("""Enter atom index > """))
                print("Calculating partial charge vs geometric variable...")
                CV_list = analysis.getCVs(1)
                analysis.charge_CV(mol, atom, CV_list[0], size, output_dir)

            elif charge_option == 4:
                print("Calculating intramolecular electrostatic potential energy...")
                energy_elec = analysis.electrostatic_energy(mol.charges, mol.coords)
                print(f"{energy_elec} kcal/mol")

        elif option_flag == 5:
            print("Comparing Datasets.")
            print("Calculating force MAE...")
            mae = 0
            for actual, prediction in zip(mol2.forces.flatten(), mol1.forces.flatten()):
                diff = prediction - actual
                mae += np.sum(abs(diff))
            mae = mae / len(mol2.forces.flatten())
            print(f"Force MAE: {mae}, kcal/mol/A")

            print("Calculating force error distribution...")
            analysis.force_MSE_dist(mol2.forces.flatten(), mol1.forces.flatten(), output_dir)

            print("Calculating force S-curve...")
            write_output.scurve(mol2.forces.flatten(), mol1.forces.flatten(),
                          output_dir, "mm_f_scurve", 1.0)
            np.savetxt(f"./{output_dir}/mm_f_test.dat", np.column_stack((
                mol2.forces.flatten(), mol1.forces.flatten())),
                       delimiter=", ", fmt="%.6f")

            print("Calculating charge MAE...")
            mae = 0
            for actual, prediction in zip(mol2.charges.flatten(), mol1.charges.flatten()):
                diff = prediction - actual
                mae += np.sum(abs(diff))
            mae = mae / len(mol2.charges.flatten())
            print(f"Charge MAE: {mae}, kcal/mol/A")

            print("Calculating charge S-curve...")
            write_output.scurve(mol2.charges.flatten(), mol1.charges.flatten(),
                                output_dir, "mm_q_scurve", 1.0)
            np.savetxt(f"./{output_dir}/mm_q_test.dat", np.column_stack((
                mol2.charges.flatten(), mol1.charges.flatten())),
                       delimiter=", ", fmt="%.6f")

            print("Calculating energy correlation with QM...")
            analysis.energy_corr(mol2.energies, mol1.energies, output_dir)

        elif option_flag == 6:
            input_dir = "md_data"
            isExist = os.path.exists(input_dir)
            if not isExist:
                print("Error - no input files in the working directory.")
                exit()
            print("Calculating 2D free energy surface...")
            analysis.fes2D(input_dir, output_dir)

    elif input_flag == 4:
        option_flag = int(input("""Generate a New Dataset...
             [1] - ...by Dihedral Rotation.
             [2] - ...by Structure Selection using Index List.
             [3] - ...by Structure Selection using RMSD Criteria.
             > """))

        if option_flag == 1:
            print("Generate a New Dataset by Dihedral Rotation.")

            input_format = int(input("""Input format:
            [1] - qm_data (.out)
            [2] - ml_data (.txt)
            [3] - md_data (.txt)
            > """))
            print()

            if input_format == 1:
                print("Input format: .out")
                input_dir = "qm_data"

            elif input_format == 2:
                print("Input format: .txt")
                input_dir = "ml_data"

            elif input_format == 3:
                print("Input format: .txt")
                input_dir = "md_data"

            else:
                print("Error - Invalid Value")
                exit()

            isExist = os.path.exists(input_dir)
            if not isExist:
                print("Error - no input files in the working directory.")
                exit()

            # read input
            mol = read_input.Molecule()
            size = 1
            init = 0
            space = 1
            print("Using first structure only...")
            if input_format == 1:
                read_input.Dataset(mol, size, init, space, input_dir, "gau")

            elif input_format == 2 or input_format == 3:
                read_input.Dataset(mol, size, init, space, input_dir, "txt")

            CV_list = analysis.getCVs(1)

            new_coords = analysis.rotate_dihedral(mol, CV_list[0])
            output_dir = "qm_data"
            isExist = os.path.exists(output_dir)
            if not isExist:
                os.makedirs(output_dir)
            print("Writing dataset...")
            write_output.gau(mol, new_coords, output_dir, True, CV_list[0])

        elif option_flag == 2:
            print("Generate a New Dataset by Structure Selection using Index List.")
            print("Input format: Text (.txt)")
            print("Output format: Text (.txt)")

            input_dir = "ml_data"
            isExist = os.path.exists(input_dir)
            if not isExist:
                print("Error - no input files in the working directory")
                exit()

            output_dir = "ml_data_new"
            isExist = os.path.exists(output_dir)
            if isExist:
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)

            while True:
                try:
                    size = int(input("Enter number of structures > "))
                    #init = int(input("Enter the initial structure > "))
                    #space = int(input("Enter spacing between structures > "))
                    init = 0
                    space = 1
                    break
                except ValueError:
                    print("Invalid Value")

            # initiate molecule class and parse dataset
            mol = read_input.Molecule()
            read_input.Dataset(mol, size, init, space, input_dir, "txt")

            # read list of structure indices TODO: put this into a module
            indices = np.loadtxt("split_indices.dat", dtype=int)
            new_energies = np.take(mol.energies, indices, axis=0)
            new_coords = np.take(mol.coords, indices, axis=0)
            new_forces = np.take(mol.forces, indices, axis=0)
            new_charges = np.take(mol.charges, indices, axis=0)
            np.savetxt(f"./{output_dir}/energies.txt", new_energies, fmt="%.10f")
            coord_file = open(f"./{output_dir}/coords.txt", "w")
            force_file = open(f"./{output_dir}/forces.txt", "w")
            charge_file = open(f"./{output_dir}/charges.txt", "w")
            for item in range(indices.shape[0]):
                for atom in range(mol.n_atom):
                    print(*new_coords[item, atom], file=coord_file)
                    print(*new_forces[item, atom], file=force_file)
                    print(new_charges[item, atom], file=charge_file)

    elif input_flag == 5:
        print("Reformat an Existing Dataset.")
        print()

        input_format = int(input("""Input format:
        [1] - qm_data (.out)
        [2] - ml_data (.txt)
        [3] - md_data (.txt)
        > """))
        print()

        output_format = int(input("""Output format:
        [1] - qm_data (.gjf)
        [2] - ml_data (.txt)
        [3] - gro_files (.gro)
        [4] - pdb_files (.pdb)
        > """))
        print()

        perm_option = "N"
        if input_format == 1 and output_format == 2:
            perm_option = str(input("Shuffle permutations? (Y/N) > "))

        while True:
            try:
                size = int(input("Enter total number of structures > "))
                init = int(input("Remove first N structures (0 for none) > "))
                space = 1 # TODO: implement this
                #space = int(input("Enter spacing between structures > "))
                break
            except ValueError:
                print("Invalid Value")

        if input_format == 1:
            print("Input format: .out")
            input_dir = "qm_data"

        elif input_format == 2:
            print("Input format: .txt")
            input_dir = "ml_data"

        elif input_format == 3:
            print("Input format: .txt")
            input_dir = "md_data"

        else:
            print("Error - Invalid Value")
            exit()

        isExist = os.path.exists(input_dir)
        if not isExist:
            print("Error - no input files in the working directory.")
            exit()

        # read input
        mol = read_input.Molecule()
        if input_format == 1:
            read_input.Dataset(mol, size, init, space, input_dir, "gau")

        elif input_format == 2 or input_format == 3:
            read_input.Dataset(mol, size, init, space, input_dir, "txt")

        if output_format == 1:
            print("Output format: .gjf")
            output_dir = "qm_data"

        elif output_format == 2:
            print("Output format: .txt")
            if perm_option == "Y":
                output_dir = "ml_data_perm"
                n_perm_grp, perm_atm, n_symm, n_symm_atm = read_input.perm(mol)
                print("Shuffling atomic permutations...")
                analysis.permute(mol, n_perm_grp, perm_atm, n_symm, n_symm_atm)
            else:
                output_dir = "ml_data"

        elif output_format == 3:
            print("Output format: .gro")
            output_dir = "gro_files"

        elif output_format == 4:
            print("Output format: .pdb")
            output_dir = "pdb_files"

        # check relevant output directory exists
        isExist = os.path.exists(output_dir)
        if isExist:
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        print("Writing dataset...")

        # write output
        if output_format == 1:
            write_output.gau(mol, mol.coords, output_dir, False, 0)
        elif output_format == 2:
            write_output.dataset(mol, output_dir)
        elif output_format == 3:
            write_output.gro(mol, output_dir)
        elif output_format == 4:
            write_output.pdb(mol, output_dir, "none")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

