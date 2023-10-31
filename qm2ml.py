import numpy as np
import random
from itertools import islice

def gau2ml(set_size, step, input_dir, output_dir, perm):
    energies = np.empty(shape=[set_size])
    errors = []

    # write .txt files
    coord_file = open(f"./{output_dir}/coords.txt", "w")
    energy_file = open(f"./{output_dir}/energies.txt", "w")
    force_file = open(f"./{output_dir}/forces.txt", "w")
    error_file = open(f"./{output_dir}/errors.txt", "w")
    charge_file = open(f"./{output_dir}/charges.txt", "w")

    # get atom count
    with open(f"./nuclear_charges.txt", "r") as nuclear_charge_file:
        n_atom = len(nuclear_charge_file.readlines())

    # read in all symmetry equivalent atoms/groups
    if perm:
        with open(f"./permutations.txt", "r") as perm_file:
            max_atm = 10
            max_symm_atm = 10
            n_perm_grp = int(perm_file.readline())
            n_symm = np.zeros(shape=[n_perm_grp], dtype=int)
            n_symm_atm = np.zeros(shape=[n_perm_grp], dtype=int)
            perm_atm = np.zeros(shape=[n_perm_grp, max_symm_atm, max_atm], dtype=int)
            for i_perm in range(0,n_perm_grp):
                n_symm[i_perm] = int(perm_file.readline())
                for i_symm in range(0,n_symm[i_perm]):
                    indices = [eval(i) for i in perm_file.readline().split()]
                    if i_symm == 0:
                        n_symm_atm[i_perm] = len(indices)
                    for i_atm in range(n_symm_atm[i_perm]):
                        perm_atm[i_perm][i_symm][i_atm] = indices[i_atm]
                        if perm_atm[i_perm][i_symm][i_atm] > n_atom:
                            print("Error - permutation atom out of range")
                            exit()
            perm_file.close()

    # set up arrays
    coord = np.empty(shape=[set_size, n_atom, 3])
    force = np.empty(shape=[set_size, n_atom, 3])
    charge = np.empty(shape=[set_size, n_atom])

    # loop over all Gaussian files, extract energies, forces and coordinates
    for i_file in range(set_size):
        if ((i_file) % step) == 0:
            normal_term = False
            qm_file = open(f"./{input_dir}/mol_{i_file+1}.out", "r")
            for line in qm_file:
                # extract atomic coordinates
                if "Input orientation:" in line:
                    coord_block = list(islice(qm_file, 4+n_atom))[-n_atom:]
                # extract energies
                if "SCF Done:" in line:
                    energies[i_file] = (float(line.split()[4]))
                # extract forces
                if "Axes restored to original set" in line:
                    force_block = list(islice(qm_file, 4+n_atom))[-n_atom:]
                # extract charges
                if "ESP charges:" in line:
                    charge_block = list(islice(qm_file, 1+n_atom))[-n_atom:]
                # assess termination state
                if "Normal termination of Gaussian 09" in line:
                    normal_term = True
                    break

            # convert to kcal/mol and print to energy.txt file
            print(energies[i_file]*627.509608, file=energy_file)

            # read atomic coordinates
            for i_atom, atom in enumerate(coord_block):
                coord[i_file, i_atom] = atom.strip('\n').split()[-3:]

            # read atomic forces
            for i_atom, atom in enumerate(force_block):
                force[i_file, i_atom] = atom.strip('\n').split()[-3:]

            # read partial charges
            for i_atom, atom, in enumerate(charge_block):
                charge[i_file, i_atom] = atom.strip('\n').split()[-1]

            # make random permutation
            if perm:
                # loop through symmetry groups
                for i_perm in range(n_perm_grp):
                    # perform 10 swap moves for this symmetry group
                    for i_swap in range(10):
                        # for this permutation randomly select a symmetry group
                        old_perm = perm_atm[i_perm][random.randint(0,n_symm[i_perm]-1)][:]
                        new_perm = perm_atm[i_perm][random.randint(0,n_symm[i_perm]-1)][:]
                        # swap and save coordinates for these groups
                        for i_atm in range(n_symm_atm[i_perm]):
                            temp_coord = np.copy(coord[i_file, old_perm[i_atm]-1])
                            coord[i_file, old_perm[i_atm]-1] = coord[i_file, new_perm[i_atm]-1]
                            coord[i_file, new_perm[i_atm]-1] = temp_coord
                            temp_force = np.copy(force[i_file, old_perm[i_atm]-1])
                            force[i_file, old_perm[i_atm]-1] = force[i_file, new_perm[i_atm]-1]
                            force[i_file, new_perm[i_atm]-1] = temp_force

            # save coordinates and forces (converting to kcal/mol/A)
            for i_atom in range(n_atom):
                print(*coord[i_file, i_atom], file=coord_file)
                print(*force[i_file, i_atom]*627.509608/0.529177, file=force_file)
                print(charge[i_file, i_atom], file=charge_file)

            if not normal_term:
                errors.append(i_file)
                print(i_file, file=error_file)
            qm_file.close()

    coord_file.close()
    energy_file.close()
    force_file.close()
    charge_file.close()
    error_file.close()

