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
            [5] - Reformat an Existing Dataset
            > """))
    except ValueError:
        print("Invalid Value")
        exit()
    except input_flag > 5:
        print("Invalid Value")
        exit()
    print()

    # determine type of calculation to do
    if input_flag == 1:

        startTime = datetime.now()
        option_flag = int(input("""Run a Molecular Dynamics Simulation.
            [1] - Use an Empirical Potential.
            [2] - Use a PairNet Potential.
            [3] - Use ANI-2x.
            > """))

        if option_flag == 1:
            print("Use an Empirical Potential.")
            potential = "empirical"
        elif option_flag == 2:
            print("Use PairNet potential.")
            potential = "pair_net"
        elif option_flag == 3:
            print("Use ANI.")
            potential = "ani"
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
        simulation, system, output_dir, md_params, gro, force = md.setup(potential, plat)

        # run simulation
        # TODO: change md simulation output to md_data?
        md.simulate(simulation, system, potential, output_dir, md_params, gro, force)

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
        read_input.Dataset(mol, size, 1, 1, input_dir, "txt")
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

        # load previously trained model
        if ann_load:
            print("Loading a trained model...")
            model = network.load(mol, ann_params)

        else:
            mol.trainval = [*range(0, n_train + n_val, 1)]
            trainval_forces = np.take(mol.forces, mol.trainval, axis=0)
            trainval_energies = np.take(mol.energies, mol.trainval, axis=0)
            # TODO: pre-scaling for charges?!
            #trainval_charges = np.take(mol.charges, mol.trainval, axis=0)
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

            # pairwise decomposition of energies
            analysis.get_eij(mol, size, output_dir1)

            # build model if not training from scratch
            if not ann_load:
                # get prescaling factors
                prescale = analysis.prescale_eij(mol, prescale)
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
            np.savetxt(f"./{output_dir2}/nuclear_charges.txt",
                (np.array(mol.atoms)).reshape(-1, 1))
            np.savetxt(f"./{output_dir2}/prescale.txt",
                (np.array(prescale)).reshape(-1, 1))

        # test model
        if ann_test:
            mol.test = [*range(n_train + n_val, size, 1)]
            if ann_load:
                analysis.get_eij(mol, size, output_dir1)

            print("Testing model...")
            network.test(model, mol, output_dir1, ann_params)

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
                    init = int(input("Enter the initial structure > "))
                    space = int(input("Enter spacing between structures > "))
                    print()
                    break
                except ValueError:
                    print("Invalid Value")

        if option_flag < 5:
            input_type = int(input("""Type of Dataset to Analyse:
            [1] - QM Dataset.
            [2] - MD Dataset.
            [3] - External Dataset.
            > """))
            print()

            if input_type == 1:
                print("Loading QM dataset...")
                input_dir = "ml_data"
                isExist = os.path.exists(input_dir)
                if not isExist:
                    print("Error - no input files detected")
                    exit()
                mol = read_input.Molecule()
                read_input.Dataset(mol, size, init, space, input_dir, "txt")

            elif input_type == 2:
                print("Loading MD dataset...")
                input_dir = "md_data"
                isExist = os.path.exists(input_dir)
                if not isExist:
                    print("Error - no input files detected")
                    exit()
                mol = read_input.Molecule()
                read_input.Dataset(mol, size, init, space, input_dir, "txt")

            elif input_type == 3:
                print("Loading external dataset...")
                input_dir = "/Users/user/datasets"
                mol = read_input.Molecule()
                read_input.Dataset(mol, size, init, space, input_dir, "ext")

        elif option_flag == 5:

            input_dir1 = "md_data"
            isExist = os.path.exists(input_dir1)
            if not isExist:
                print("Error - no input files detected")
                exit()
            mol1 = read_input.Molecule()
            read_input.Dataset(mol1, size, init, space, input_dir1, "txt")

            input_dir2 = "qm_data"
            isExist = os.path.exists(input_dir2)
            if not isExist:
                print("Error - no input files detected")
                exit()
            mol2 = read_input.Molecule()
            read_input.Dataset(mol1, size, init, space, input_dir1, "txt")

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
            analysis.check_stability(mol, size, init, space, output_dir)

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
                CV_list = analysis.getCVs(1)
                analysis.energy_CV(mol, CV_list[0], size, output_dir)

            elif geom_flag == 2:
                print("Get root mean squared deviation of distance matrix.")
                rmsd_dist = analysis.rmsd_dist(mol, size)
                print(f"Distance matrix RMSD: {np.mean(rmsd_dist)} Angstrom")

            elif geom_flag == 3:
                print("Get 1D probability distribution of geometric variable.")
                n_bins = int(input("Enter the number of bins > "))
                CV_list = analysis.getCVs()
                analysis.pop1D(mol, n_bins, CV_list[0], output_dir, init, size)

            elif geom_flag == 4:
                print("Get 2D probability distribution of geometric variable.")
                n_bins = int(input("Enter the number of bins > "))
                CV_list = analysis.getCVs()
                analysis.pop2D(mol, n_bins, CV_list, output_dir, init, size)

        elif option_flag == 4:
            print("Analyse Charges.")
            charge_option = int(input("""
                [1] Calculate mean partial charges.
                [2] Calculate partial charge probability distribution.
                [3] Calculate partial charge vs geometric variable.
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

        elif option_flag == 5:
            print("Compare Datasets.")

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
                          output_dir, "mm_f_scurve")
            np.savetxt(f"./{output_dir}/mm_f_test.dat", np.column_stack((
                mol2.forces.flatten(), mol1.forces.flatten())),
                       delimiter=", ", fmt="%.6f")

            print("Calculating energy correlation with QM...")
            analysis.energy_corr(mol2.energies, mol1.energies, output_dir)

        elif option_flag == 6:
            print("Calculating 2D free energy surface...")
            analysis.fes2D(input_dir, output_dir)
            input_dir = "md_data"
            isExist = os.path.exists(input_dir)
            if not isExist:
                print("Error - no input files detected")
                exit()

    elif input_flag == 4:
        print("Generate a Dataset.")
        option_flag = int(input("""Generate a New Dataset...
             [1] - ...by Dihedral Rotation.
             [2] - ...by Structure Selection using Index List.
             [3] - ...by Structure Selection using RMSD Criteria.
             > """))

        if option_flag == 1:
            print("Generate a New Dataset by Dihedral Rotation.")
            size, init, space, opt_prop = 1

            print("Input format: Gaussian (.out)")
            print("Output format: Gaussian (.gjf)")

            input_dir = "qm_data"
            isExist = os.path.exists(input_dir)
            if not isExist:
                print("Error - no input files detected")
                exit()
            output_dir = input_dir

            mol = read_input.Molecule()
            read_input.Dataset(mol, size, init, space, input_dir, "gau")

            CV_list = analysis.getCVs()
            if len(CV_list) > 1:
                print("Error - number of collective variables cannot be > 1")
                exit()

            new_coords= analysis.rotate_dihedral(mol, CV_list)
            write_output.gau(mol, new_coords, output_dir, opt_prop, CV_list)

        elif option_flag == 2:
            print("Generate a New Dataset by Structure Selection using Index List.")
            print("Input format: Text (.txt)")
            print("Output format: Text (.txt)")

            input_dir = "ml_data"
            isExist = os.path.exists(input_dir)
            if not isExist:
                print("Error - no input files detected")
                exit()

            output_dir = "ml_data_new"
            isExist = os.path.exists(output_dir)
            if isExist:
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)

            while True:
                try:
                    size = int(input("Enter number of structures > "))
                    init = int(input("Enter the initial structure > "))
                    space = int(input("Enter spacing between structures > "))
                    break
                except ValueError:
                    print("Invalid Value")

            # initiate molecule class and parse dataset
            mol = read_input.Molecule()
            read_input.Dataset(mol, size, init, space, input_dir, "txt")

            # read list of structure indices
            indices = np.loadtxt("split_indices.dat", dtype=int)
            new_energies = np.take(mol.energies, indices, axis=0)
            new_coords = np.take(mol.coords, indices, axis=0)
            new_forces = np.take(mol.forces, indices, axis=0)
            new_charges = np.take(mol.charges, indices, axis=0)
            np.savetxt(f"./{output_dir}/energies.txt", new_energies, fmt="%.10f")
            coord_file = open(f"./{output_dir}/coords.txt", "w")
            force_file = open(f"./{output_dir}/forces.txt", "w")
            charge_file = open(f"./{output_dir}/forces.txt", "w")
            for item in range(indices.shape[0]):
                for atom in range(mol.n_atom):
                    print(*new_coords[item, atom], file=coord_file)
                    print(*new_forces[item, atom], file=force_file)
                    print(*new_charges[item, atom], file=charge_file)

    elif input_flag == 5:
        print("Reformat an Existing Dataset.")

        input_format = int(input("""Input format:
        [1] - Gaussian (.out) - qm_data directory required.
        [2] - Text (.txt) - ml_data directory required.
        [3] - Gromacs (.gro) - md_data directory required.
        >"""))

        output_format = int(input("""Output format:
        [1] - Gaussian (.gjf)
        [2] - Text (.txt)
        [3] - Gromacs (.gro)
        >"""))

        if input_format == output_format:
            print("ERROR - input and output format are the same. Nothing to do.")
            exit()

        while True:
            try:
                size = int(input("Enter number of structures > "))
                init = int(input("Enter the initial structure > "))
                space = int(input("Enter spacing between structures > "))
                break
            except ValueError:
                print("Invalid Value")

        mol = read_input.Molecule()
        if input_format == 1:
            print("Input format: Gaussian (.out)")
            input_dir = "qm_data"

        elif input_format == 2:
            print("Input format: Text (.txt)")
            input_dir = "ml_data"

        elif input_format == 3:
            print("Input format: Gromacs (.gro)")
            input_dir = "md_data"

        isExist = os.path.exists(input_dir)
        if not isExist:
            print("Error - no input files detected")
            exit()

        # read input
        if input_format == 1:
            read_input.Dataset(mol, size, init, space, input_dir, "gau")
        elif input_format == 2 or input_format == 3:
            read_input.Dataset(mol, size, init, space, input_dir, "txt")

        if output_format == 1:
            print("Output format: Gaussian (.gjf)")
            output_dir = "qm_data"

        elif output_format == 2:
            print("Output format: Text (.txt)")
            perm_option = str(input("Shuffle permutations? (Y/N) > "))
            if perm_option == "Y":
                perm = True
            else:
                perm = False
            if perm:
                output_dir = "ml_data_perm"
                mol = read_input.Molecule()
                read_input.Dataset(mol, size, init, space, input_dir, "txt")
                n_perm_grp, perm_atm, n_symm, n_symm_atm = read_input.perm(mol)
                print("Shuffling atomic permutations...")
                analysis.permute(mol, n_perm_grp, perm_atm, n_symm, n_symm_atm)
            else:
                output_dir = "ml_data"

        elif output_format == 3:
            print("Output format: Gromacs (.gro)")
            output_dir = "md_data"

        # check relevant output directory exists
        isExist = os.path.exists(output_dir)
        if not isExist:
            os.makedirs(output_dir)

        # write output
        if output_format == 1:
            write_output.gau(mol, mol.coords, output_dir, 0)
        elif output_format == 2:
            write_output.dataset(mol, output_dir)
        elif output_format == 3:
            write_output.gro(mol, output_dir)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

