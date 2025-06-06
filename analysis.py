import numpy as np
import write_output
from scipy.stats import binned_statistic
import random, math

def dist(mol, set_size, output_dir):
    n_atom = len(mol.atoms)
    hist, bin = np.histogram(mol.forces.flatten(),200,(-250,250))
    bin = bin[range(1, bin.shape[0])]
    bin_width = bin[1] - bin[0]
    write_output.lineplot(bin, hist / bin_width / set_size / 3.0 / n_atom, "linear",
                "force (kcal/mol/A)", "probability", "force_dist", output_dir)
    np.savetxt(f"./{output_dir}/force_dist.dat",
        np.column_stack((bin, hist / bin_width / set_size / 3.0 / n_atom)),
        delimiter = " ",fmt="%.6f")
    hist, bin = np.histogram(mol.energies, 50, (np.min(mol.energies),np.max(mol.energies)))
    bin = bin[range(1, bin.shape[0])]
    bin_width = bin[1] - bin[0]
    write_output.lineplot(bin, hist / bin_width / set_size, "linear", "energy",
        "probability", "energy_dist", output_dir)
    np.savetxt(f"./{output_dir}/energy_dist.dat",
        np.column_stack((bin, hist / bin_width / set_size)), delimiter = " ",
        fmt="%.6f")
    return None


def charge_dist(mol, index, set_size, output_dir):
    charge = np.empty(shape=[set_size])
    for item in range(set_size):
        charge[item] = mol.charges[item][index]
    hist, bin = np.histogram(charge,200,(np.min(charge),np.max(charge)))
    bin = bin[range(1, bin.shape[0])]
    bin_width = bin[1] - bin[0]
    write_output.lineplot(bin, hist / bin_width / set_size, "linear",
                "charge (e)", "probability", "charge_dist", output_dir)
    np.savetxt(f"./{output_dir}/charge_dist_{index}.dat",
        np.column_stack((bin, hist / bin_width / set_size)),
        delimiter = " ",fmt="%.6f")
    return None


def charge_CV(mol, index, atom_indices, set_size, output_dir):
    partial_charge = mol.charges[:,index]
    CV_list = atom_indices
    CV = np.empty(shape=[set_size])
    for item in range(set_size):
        p = np.zeros([len(CV_list), 3])
        p[0:] = mol.coords[item][CV_list[:]]
        if len(CV_list) == 2:
            x_label = "$r_{ij} / \AA$"
            CV[item] = distance(p)
        elif len(CV_list) == 3:
            x_label = "$\u03F4_{ijk}  (degrees)$"
            CV[item] = angle(p)
        elif len(CV_list) == 4:
            x_label = "$\u03C6_{ijkl} (degrees)$"
            CV[item] = dihedral(p)

    write_output.scatterplot(CV, partial_charge, "linear", x_label,
        "partial charge (e)", f"charge_geom_scatter_{index}", output_dir)
    np.savetxt(f"./{output_dir}/charge_geom_scatter_{index}.dat",
        np.column_stack((CV, partial_charge)), delimiter=" ", fmt="%.6f")

    means, edges, counts = binned_statistic(CV, partial_charge,
        statistic='mean', bins=18, range=(-180.0, 180.0)) #TODO: set mean and max sensibly
    bin_width = edges[1] - edges[0]
    bin_centers = edges[1:] - bin_width / 2
    write_output.lineplot(bin_centers, means, "linear", x_label,
        "mean partial charge (e)", f"charge_geom_{index}", output_dir)
    np.savetxt(f"./{output_dir}/charge_geom_{index}.dat",
        np.column_stack((bin_centers, means)), delimiter=" ", fmt="%.6f")
    return None


def energy_CV(mol, n_bins, atom_indices, set_size, output_dir):
    CV_list = atom_indices
    CV = np.empty(shape=[set_size])
    for item in range(set_size):
        p = np.zeros([len(CV_list), 3])
        p[0:] = mol.coords[item][CV_list[:]]
        if len(CV_list) == 2:
            x_label = "$r_{ij} / \AA$"
            CV[item] = distance(p)
        elif len(CV_list) == 3:
            x_label = "$\u03F4_{ijk}  (degrees)$"
            CV[item] = angle(p)
        elif len(CV_list) == 4:
            x_label = "$\u03C6_{ijkl} (degrees)$"
            CV[item] = dihedral(p)
    # plot distribution, scatter and save data
    energy = mol.energies - np.min(mol.energies)
    write_output.scatterplot(CV, energy, "linear", x_label,
        "rel. energy (kcal/mol)", "energy_geom_scatter", output_dir)
    np.savetxt(f"./{output_dir}/energy_geom_scatter.dat",
        np.column_stack((CV, energy)), delimiter=" ", fmt="%.6f")

    if len(CV_list) == 2 or len(CV_list) == 3:
        range_min = np.min(CV)
        range_max = np.max(CV)
    else:
        range_min = -180.0
        range_max = 180.0
    means, edges, counts = binned_statistic(CV, energy, statistic='min',
        bins=n_bins, range=(range_min, range_max))
    bin_width = edges[1] - edges[0]
    bin_centers = edges[1:] - bin_width / 2
    write_output.lineplot(bin_centers, means, "linear", x_label,
        "rel. mean energy (kcal/mol)", "energy_geom_dist", output_dir)
    np.savetxt(f"./{output_dir}/energy_geom_dist.dat",
        np.column_stack((bin_centers, means)), delimiter = " ",
               fmt="%.6f")
    return None

def rmsd_dist(mol, set_size):
    n_atoms = len(mol.atoms)
    _NC2 = int(n_atoms * (n_atoms - 1) / 2)
    r_ij_0 = np.zeros((n_atoms, n_atoms))
    rmsd_dist = np.zeros(set_size)
    # loop over all structures
    for s in range(set_size):
        sum_rmsd_dist = 0
        # loop over all atom pairs
        for i in range(n_atoms):
            for j in range(i):
                r_ij = np.linalg.norm(mol.coords[s][i] - mol.coords[s][j])
                if s == 0:
                    r_ij_0[i,j] = r_ij
                else:
                    rij_diff = r_ij - r_ij_0[i,j]
                    sum_rmsd_dist += rij_diff**2
        if s != 0:
            rmsd_dist[s] = np.sqrt(sum_rmsd_dist / n_atoms / n_atoms)
    return rmsd_dist

def prescale_e(mol, energies, forces, norm_scheme):
    if norm_scheme == "z-score":
        mu = np.mean(energies)
        sigma = np.std(energies)
        prescale =[mu, sigma, 0, 0, 0, 0]
        mol.energies = (mol.orig_energies - mu) / sigma
    elif norm_scheme == "force":
        min_e, max_e = np.min(energies), np.max(energies)
        min_f, max_f = np.min(forces), np.max(forces)
        min_f = np.min(np.abs(forces))
        prescale = [min_e, max_e, min_f, max_f, 0, 0]
        mol.energies = ((max_f-min_f)*(mol.orig_energies-min_e)/(max_e-min_e)+min_f)
    elif norm_scheme == "none":
        prescale = [0, 0, 0, 0, 0, 0]
        mol.energies = mol.orig_energies
    return prescale

def prescale_eij(mol, prescale):
    n_atoms = len(mol.atoms)
    n_pairs = int(n_atoms * (n_atoms - 1) / 2)
    input_NRF = mol.mat_NRF.reshape(-1, n_pairs)
    trainval_input_NRF = np.take(input_NRF, mol.trainval, axis=0)
    trainval_output_eij = np.take(mol.output_eij, mol.trainval, axis=0)
    prescale[4] = np.max(np.abs(trainval_input_NRF))
    prescale[5] = np.max(np.abs(trainval_output_eij))
    return prescale

def get_eij(mol, set_size, output_dir):
    '''Get decomposed energies and forces from the same simultaneous equation'''

    n_atoms = len(mol.atoms)
    _NC2 = int(n_atoms * (n_atoms - 1) / 2)

    # assign arrays
    mol.mat_NRF = np.zeros((set_size, _NC2))
    mol.mat_r = np.zeros((set_size, _NC2))
    bias = np.zeros((set_size, _NC2))
    mol.output_eij = np.zeros((set_size, _NC2))
    mol.mat_i = np.zeros(_NC2)
    mol.mat_j = np.zeros(_NC2)

    # loop over all structures
    for s in range(set_size):
        _N = -1
        # calculate the distance matrix, r_ij
        for i in range(n_atoms):
            zi = mol.atoms[i]
            for j in range(i):
                _N += 1
                zj = mol.atoms[j]

                if s == 0:
                    mol.mat_i[_N] = i
                    mol.mat_j[_N] = j

                # calculate interatomic distances, save to distance matrix
                r_ij = np.linalg.norm(mol.coords[s][i] - mol.coords[s][j])
                mol.mat_r[s, _N] = r_ij

                # calculate interatomic nuclear repulsion force (input features)
                mol.mat_NRF[s, _N] = get_NRF(zi, zj, r_ij)
                bias[s, _N] = 1 / r_ij

        # calculation normalisation factor, N
        norm_recip_r = 1 / (np.sum(bias[s] ** 2) ** 0.5)

        # normalise  pair energy biases to give dimensionless quantities
        e_ij = bias[s].reshape((1, _NC2)) * norm_recip_r

        # reference energy biases, will be predicted by the trained potential
        mol.energies = np.vstack(mol.energies)
        mol.output_eij[s] = np.matmul(np.linalg.pinv(e_ij), mol.energies[s])

    # flatten output_matFE instead below?
    write_output.scatterplot([mol.mat_r.flatten()], [mol.output_eij.flatten()], "linear",
        "$r_{ij}$ / $\AA$", "$e_{ij}$ / kcal/mol", "eij_rij", output_dir)
    np.savetxt(f"./{output_dir}/eij_rij.dat",
        np.column_stack((mol.mat_r.flatten(), mol.output_eij.flatten())),
        delimiter=" ", fmt="%.6f")
    hist, bin = np.histogram(mol.output_eij.flatten(), 200,
        (np.min(mol.output_eij.flatten()), np.max(mol.output_eij.flatten())))
    bin = bin[range(1, bin.shape[0])]
    bin_width = bin[1] - bin[0]
    write_output.lineplot(bin, hist / bin_width / _NC2 / set_size, "linear",
        "$e_{ij}$ / kcal/mol", "probability", "eij_dist", output_dir)
    np.savetxt(f"./{output_dir}/eij_dist.dat",
               np.column_stack((bin, hist / bin_width / _NC2 / set_size)),
               delimiter=" ", fmt="%.6f")
    return None


# TODO: this function will be removed.
def get_NRF(zA, zB, r):
    _NRF = r and (zA * zB * np.float64(627.5095 * 0.529177) / (r ** 2))
    return _NRF


# TODO: can this be removed?
def z_score(x, prescale):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # normalise energies
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x - mu) / sigma
    return x_norm, mu, sigma


def pop2D(mol, n_bins, CV_list, output_dir, set_size):
    bin_width = 360 / n_bins
    pop = np.zeros(shape=(n_bins, n_bins))
    for item in range(set_size):
        bin = np.empty(shape=[CV_list.shape[0]], dtype=int)
        for i_dih in range(CV_list.shape[0]):
            p = np.zeros([CV_list.shape[1], 3])
            p[0:] = mol.coords[item][CV_list[i_dih][:]]
            bin[i_dih] = int((dihedral(p) + 180) / bin_width)
            if bin[i_dih] == n_bins:  # this deals with 360 degree angles
                bin[i_dih] = 0
        pop[bin[1]][bin[0]] += 1
    pop = pop / (set_size)
    #pop[pop==0] = np.nan
    x, y = np.meshgrid(np.linspace(-180, 180, n_bins), np.linspace(-180, 180, n_bins))
    write_output.heatmap2D(x, y, pop, output_dir, "pop_2d", "gist_heat", 1)
    count = 0
    for i in range(n_bins):
        for j in range(n_bins):
            if pop[i][j] != 0:
                count += 1
    print("% of surface populated:", 100*count /(n_bins*n_bins))
    return None


def pop3D(mol, n_bins, CV_list, set_size):
    bin_width = 360 / n_bins
    pop = np.zeros(shape=(n_bins, n_bins, n_bins))
    for item in range(set_size):
        bin = np.empty(shape=[CV_list.shape[0]], dtype=int)
        for i_dih in range(CV_list.shape[0]):
            p = np.zeros([CV_list.shape[1], 3])
            p[0:] = mol.coords[item][CV_list[i_dih][:]]
            bin[i_dih] = int((dihedral(p) + 180) / bin_width)
            if bin[i_dih] == n_bins:  # this deals with 360 degree angles
                bin[i_dih] = 0
        pop[bin[2]][bin[1]][bin[0]] += 1
    pop = pop / (set_size)
    count = 0
    for i in range(n_bins):
        for j in range(n_bins):
            for k in range(n_bins):
                if pop[i][j][k] != 0:
                    count += 1
    print("% of surface populated:", 100*count /(n_bins*n_bins*n_bins))
    return None

def pop4D(mol, n_bins, CV_list, set_size):
    bin_width = 360 / n_bins
    pop = np.zeros(shape=(n_bins, n_bins, n_bins, n_bins))
    for item in range(set_size):
        bin = np.empty(shape=[CV_list.shape[0]], dtype=int)
        for i_dih in range(CV_list.shape[0]):
            p = np.zeros([CV_list.shape[1], 3])
            p[0:] = mol.coords[item][CV_list[i_dih][:]]
            bin[i_dih] = int((dihedral(p) + 180) / bin_width)
            if bin[i_dih] == n_bins:  # this deals with 360 degree angles
                bin[i_dih] = 0
        pop[bin[3]][bin[2]][bin[1]][bin[0]] += 1
    pop = pop / (set_size)
    count = 0
    for i in range(n_bins):
        for j in range(n_bins):
            for k in range(n_bins):
                for l in range(n_bins):
                    if pop[i][j][k][l] != 0:
                        count += 1
    print("% of surface populated:", 100*count /(n_bins*n_bins*n_bins*n_bins))
    return None


def pop1D(mol, n_bins, CV_list, output_dir, set_size):
    CV = np.empty(shape=[set_size])
    for item in range(set_size):
        p = np.zeros([len(CV_list), 3])
        p[0:] = mol.coords[item][CV_list[:]]
        if len(CV_list) == 2:
            x_label = "$r_{ij} / \AA$"
            CV[item] = distance(p)
        elif len(CV_list) == 3:
            x_label = "$\u03F4_{ijk}  (degrees)$"
            CV[item] = angle(p)
        elif len(CV_list) == 4:
            x_label = "$\u03C6_{ijkl} (degrees)$"
            CV[item] = dihedral(p)
    if len(CV_list) == 2 or len(CV_list) == 3:
        range_min = np.min(CV)
        range_max = np.max(CV)
    else:
        range_min = -180.0
        range_max = 180.0
    hist, bin = np.histogram(CV, n_bins, (range_min, range_max))
    bin = bin[range(1, bin.shape[0])]
    write_output.lineplot(bin, hist / (set_size), "linear", x_label,
    "probability", "geom_dist", output_dir)
    np.savetxt(f"./{output_dir}/geom_dist.dat", np.column_stack((bin,
        hist / (set_size))), delimiter=" ", fmt="%.6f")


def force_MSE_dist(baseline, values, output_dir):
    RSE = np.sqrt((baseline - values)**2)
    MSE = binned_statistic(baseline, RSE, statistic='mean',
        bins=50, range=(-200.0, 200.0))
    bin_width = (MSE.bin_edges[1] - MSE.bin_edges[0])
    bin_centers = MSE.bin_edges[1:] - bin_width / 2
    write_output.lineplot(bin_centers, MSE.statistic, "linear",
        "QM force (kcal/mol/$\AA$)", "MSE (kcal/mol/$\AA$)", "force_MSE_dist", output_dir)
    np.savetxt(f"./{output_dir}/force_mse_dist.dat", np.column_stack((bin_centers,
        MSE.statistic)), delimiter = " ", fmt="%.6f")
    write_output.scatterplot(baseline, RSE, "linear", "QM force (kcal/mol/$\AA$)",
        "RSE (kcal/mol/$\AA$)", "force_error_scatter", output_dir)
    np.savetxt(f"./{output_dir}/force_error_dist.dat", np.column_stack((baseline,
        RSE)), delimiter = " ", fmt="%.6f")
    return None


def energy_corr(baseline, values, output_dir):
    set_size = len(baseline)
    rel_baseline = baseline - np.min(baseline)
    rel_values = values - np.min(values)
    hist, bin = np.histogram(values, 50, (np.min(values), np.max(values)))
    bin = bin[range(1, bin.shape[0])]
    bin_width = bin[1] - bin[0]
    write_output.lineplot(bin, hist / bin_width / set_size, "linear", "energy",
        "probability", "mm_energy_dist", output_dir)
    np.savetxt(f"./{output_dir}/mm_energy_dist.dat", np.column_stack((bin,
        hist / bin_width / set_size)), delimiter=" ", fmt="%.6f")
    write_output.scatterplot(rel_baseline, rel_values, "linear", "Rel. QM energy (kcal/mol)",
        "Rel. MD energy (kcal/mol)", "energy_error_scatter", output_dir)
    return None


def getCVs(n_CV):
    for i_CV in range(n_CV):
        atom_indices = input(f"""
        Enter atom indices separated by spaces:
            e.g. for a distance "0 1"
            e.g. for an angle "1 2 3 4"
            e.g. for a dihedral "5 4 6 10"
            Consult mapping.dat for connectivity.
        > """)
        if i_CV == 0:
            CV_list = np.empty(shape=[n_CV, len(atom_indices.split())], dtype=int)
        CV_list[i_CV, :] = np.array(atom_indices.split())
    return CV_list

def distance(p):
    b = p[:-1] - p[1:]
    return np.sqrt(np.sum(np.square(b)))

def angle(p):
    b = p[:-1] - p[1:]
    x = np.dot(-b[0], b[1]) / np.linalg.norm(b[0]) / np.linalg.norm(b[1])
    return np.degrees(np.arccos(x))

def dihedral(p):
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array( [ v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0], b[2]] ] )
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1,1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.degrees(np.arctan2( y, x ))

def check_stability(mol, set_size, output_dir):
    # calculate bond distance list for equilibrium structure
    n_atoms = len(mol.atoms)
    max_r = 1.40
    max_bonds = n_atoms * 3
    atom_indices = np.zeros([max_bonds, 2], dtype=int)
    bond_dist = np.zeros([max_bonds], dtype=float)
    n_bonds = 0
    for i in range(n_atoms):
        for j in range(i):
            r_ij = np.linalg.norm(mol.coords[0][i] - mol.coords[0][j])
            if r_ij < max_r:
                atom_indices[n_bonds][0] = i
                atom_indices[n_bonds][1] = j
                bond_dist[n_bonds] = r_ij
                n_bonds += 1

    # open file for writing split indices
    stable_list = open(f"./{output_dir}/stable_structures.dat", "w")

    print("Checking stability...")
    max_dev = 0.2 # maximum bond distance deviation
    min_dist = 0.85 # minimum interatomic distance
    stable_trj = True
    stable_count = 0
    for s in range(set_size):

        stable_structure = True

        for i_bond in range(n_bonds):
            p = np.zeros([2, 3])
            p[:] = mol.coords[s][atom_indices[i_bond][:]]
            r_ij = distance(p)
            if abs(r_ij - bond_dist[i_bond]) > max_dev:
                print("frame, bond number, atom i, atom j, bond dist (A), ref dist (A)")
                print(s, i_bond, atom_indices[i_bond][0],
                    atom_indices[i_bond][1], r_ij, bond_dist[i_bond])
                print("Writing .pdb file...")
                write_output.pdb(mol, output_dir, s)
                stable_structure = False

        for j in range(i):
            r_ij = np.linalg.norm(mol.coords[s][i] - mol.coords[s][j])
            if r_ij < min_dist:
                print("frame, atom i, atom j, pair dist (A)")
                print(s, i, j, r_ij)
                print("Writing .pdb file...")
                write_output.pdb(mol, output_dir, s)
                stable_structure = False

        if stable_structure == True:
            stable_count += 1
            stable_list.write(f"{s}\n")
        else:
            stable_trj = False

    if stable_trj:
        print("Trajectory is stable - no unphysical structures generated.")
    else:
        print("Trajectory is unstable - unphysical structures generated.")
        print(f"{set_size - stable_count} structures removed")

    return None


def permute(mol, n_perm_grp, perm_atm, n_symm, n_symm_atm):
    for i_file in range(len(mol.energies)):
        # loop through symmetry groups
        for i_perm in range(n_perm_grp):
            # perform 10 swap moves for this symmetry group
            for i_swap in range(10):
                # for this permutation randomly select a symmetry group
                old_perm = perm_atm[i_perm][random.randint(0,n_symm[i_perm]-1)][:]
                new_perm = perm_atm[i_perm][random.randint(0,n_symm[i_perm]-1)][:]
                # swap and save coordinates for these groups
                for i_atm in range(n_symm_atm[i_perm]):
                    temp_coord = np.copy(mol.coords[i_file, old_perm[i_atm]-1])
                    mol.coords[i_file, old_perm[i_atm]-1] = \
                        mol.coords[i_file, new_perm[i_atm]-1]
                    mol.coords[i_file, new_perm[i_atm]-1] = temp_coord
                    temp_force = np.copy(mol.forces[i_file, old_perm[i_atm]-1])
                    mol.forces[i_file, old_perm[i_atm]-1] = \
                        mol.forces[i_file, new_perm[i_atm]-1]
                    mol.forces[i_file, new_perm[i_atm]-1] = temp_force
                    temp_charge = np.copy(mol.charges[i_file, old_perm[i_atm]-1])
                    mol.charges[i_file, old_perm[i_atm] - 1] = \
                        mol.charges[i_file, new_perm[i_atm] - 1]
                    mol.charges[i_file, new_perm[i_atm] - 1] = temp_charge


def rotate_dihedral(mol, CV_list):

    atom_indices = input(f"""
         Enter indices of atoms to rotate:
         e.g. "6 10 11"
         Consult mapping.dat for connectivity.
         > """)
    rot_list = [eval(i_rot) for i_rot in atom_indices.split()]

    spacing = int(input("Enter the rotation interval (degrees) > "))
    set_size = int(360 / spacing)
    CV = np.empty(shape=[set_size])
    p = np.zeros([len(CV_list), 3])
    p[0:] = mol.coords[0][CV_list[:]]
    CV[0] = dihedral(p)
    print(f"Initial torsion angle = {CV[0]:.1f} degrees")
    axis = (p[2] - p[1]) / np.linalg.norm(p[2] - p[1])

    coord = np.empty(shape=[set_size, mol.n_atom, 3])
    # loop through all structures
    for i_angle in range(set_size):
        coord[i_angle] = mol.coords[0]
        # determine rotation angle for this structure (radians)
        # add option to do reverse scan (times by -1)
        angle = (i_angle * spacing) * np.pi / 180
        # generate rotation matrix for this structure
        mat_rot = generate_rotation_matrix(angle, axis)
        # loop through atoms to be rotated
        for i_atm in range(len(rot_list)):
            # shift to new origin
            old_coords = coord[i_angle][rot_list[i_atm]][:] - coord[i_angle][CV_list[2]][:]
            # rotate old coordinates using rotation matrix and shift to old origin
            coord[i_angle][rot_list[i_atm]][:] = np.matmul(mat_rot, old_coords) + coord[i_angle][CV_list[2]][:]
        p[0:] = coord[i_angle][CV_list[:]]

    return coord

def generate_rotation_matrix(angle, axis):
    #https://en.wikipedia.org/wiki/Rotation_matrix
    #"Rotation matrix from axis and angle"
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    rotation = R.from_rotvec(angle * np.array(axis))
    return rotation.as_matrix()


def get_interatomic_charges(self):
    # TODO: remove all this, not used
    '''
    This function takes molecule coords (C), atomic charges (Q) and
    nuclear charges (Z) and decomposes atomic charges with a 1/r bias.
    Decomposed pairwise charges are recomposed and checked with
    original atomic charges to ensure the decomposition scheme is
    performed correctly.

    Variables are -
    self:       molecule object containing ZCFEQ information
    '''

    n_atoms = len(self.atoms)
    _NC2 = int(n_atoms * (n_atoms - 1) / 2)
    n_structures = len(self.coords)
    mat_bias = np.zeros((n_structures, n_atoms, _NC2))
    mat_bias2 = np.zeros((n_structures, _NC2))
    mat_Q = []
    for s in range(n_structures):
        _N = -1
        for i in range(n_atoms):
            for j in range(i):
                _N += 1
                r_ij = np.linalg.norm(self.coords[s][i] - self.coords[s][j])
                if i != j:
                    bias = 1 / r_ij
                    mat_bias[s, i, _N] = bias
                    mat_bias[s, j, _N] = -bias
                    mat_bias2[s, _N] = bias

        charges2 = self.charges[s].reshape(n_atoms)
        _Q = np.matmul(np.linalg.pinv(mat_bias[s]), charges2)
        _N2 = -1
        for i in range(n_atoms):
            for j in range(i):
                _N2 += 1
                _Q[_N2] = _Q[_N2] * mat_bias2[s, _N2]
        mat_Q.append(_Q)

    self.mat_Q = np.reshape(np.vstack(mat_Q), (n_structures, _NC2))
    recomp_Q = get_recomposed_charges(self.coords, self.mat_Q, n_atoms, _NC2)

    if np.array_equal(np.round(recomp_Q, 1),
                      np.round(self.charges, 1)) == False:
        raise ValueError('Recomposed charges {} do not ' \
                         'equal initial charges {}'.format(
            recomp_Q, self.charges))


def get_recomposed_charges(all_coords, all_prediction, n_atoms, _NC2):
    '''Convert pairwise decomposed charges back into atomic charges.'''
    all_recomp_charges = []
    for coords, prediction in zip(all_coords, all_prediction):
        qij = np.zeros((n_atoms, n_atoms))
        # normalised interatomic vectors
        q_list = []
        for i in range(1, n_atoms):
            for j in range(i):
                # TODO: where is the 1/r bias???
                qij[i, j] = 1
                qij[j, i] = -qij[i, j]
                q_list.append([i, j])
        _T = np.zeros((n_atoms, _NC2))
        for i in range(int(_T.shape[0])):
            for k in range(len(q_list)):
                if q_list[k][0] == i:
                    _T[range(i, (i + 1)), k] = qij[q_list[k][0], q_list[k][1]]
                if q_list[k][1] == i:
                    _T[range(i, (i + 1)), k] = qij[q_list[k][1], q_list[k][0]]
        recomp_charges = np.dot(_T, prediction.flatten())
        all_recomp_charges.append(recomp_charges)
    return np.array(all_recomp_charges).reshape(-1, n_atoms)


def electrostatic_energy(charges, coords):
    elec = np.zeros(charges.shape[0])
    for s in range(charges.shape[0]):
        for i in range(charges.shape[1]):
            for j in range(i):
                r_ij = np.linalg.norm(coords[s][i] - coords[s][j])
                coul_sum = charges[s][i] * charges[s][j] / r_ij
                elec[s] = elec[s] + coul_sum
    # converts to kcal/mol
    return (elec*(1.0e10)*(6.022e23)*(1.602e-19)**2)/4.184/1000/8.854e-12


def forceerror2D(coords, CV_list, output_dir, ref, pred):
    n_bins = 18
    bin_width = 360 / n_bins
    pop = np.zeros(shape=(n_bins, n_bins))
    n_atom = coords.shape[1]
    bin_tot = np.zeros(shape=(n_bins, n_bins))
    for item in range(len(coords)):
        bin = np.empty(shape=[CV_list.shape[0]], dtype=int)
        for i_dih in range(CV_list.shape[0]):
            p = np.zeros([CV_list.shape[1], 3])
            p[0:] = coords[item][CV_list[i_dih][:]]
            bin[i_dih] = int((dihedral(p) + 180) / bin_width)
        for atom in range(n_atom):
            for force in range(3):
                pop[bin[1]][bin[0]] += 1
                error = abs(ref[item][atom][force]-pred[item][atom][force])
                bin_tot[bin[1]][bin[0]] += error
    with np.errstate(divide='ignore'): #TODO: why doesnt this work?
        bin_error = bin_tot / pop
    x, y = np.meshgrid(np.linspace(-180, 180, n_bins),
        np.linspace(-180, 180, n_bins))
    write_output.heatmap2D(x, y, bin_error, output_dir, "error_2d", "cividis", 2)
    return None


def energyerror2D(coords, CV_list, output_dir, ref, pred):
    n_bins = 18
    bin_width = 360 / n_bins
    pop = np.zeros(shape=(n_bins, n_bins))
    bin_tot = np.zeros(shape=(n_bins, n_bins))
    for item in range(len(coords)):
        bin = np.empty(shape=[CV_list.shape[0]], dtype=int)
        for i_dih in range(CV_list.shape[0]):
            p = np.zeros([CV_list.shape[1], 3])
            p[0:] = coords[item][CV_list[i_dih][:]]
            bin[i_dih] = int((dihedral(p) + 180) / bin_width)
        pop[bin[1]][bin[0]] += 1
        error = abs(ref[item] - pred[item])
        bin_tot[bin[1]][bin[0]] += error
    with np.errstate(divide='ignore'): #TODO: why doesnt this work?
        bin_error = bin_tot / pop
    x, y = np.meshgrid(np.linspace(-180, 180, n_bins),
        np.linspace(-180, 180, n_bins))
    write_output.heatmap2D(x, y, bin_error, output_dir, "error_2d", "plasma", 2)
    return None


def get_rij(coords, n_atom, size):
    '''
    returns the distance matrix for a structure or set of structures
    '''

    n_pair = int(n_atom * (n_atom - 1) / 2)
    mat_r = np.zeros((size, n_pair))
    # loop over all structures
    for s in range(size):
        _N = -1
        # calculate the distance matrix, r_ij
        for i in range(n_atom):
            for j in range(i):
                _N += 1
                # calculate interatomic distances, save to distance matrix
                r_ij = np.linalg.norm(coords[s][i] - coords[s][j])
                mat_r[s, _N] = r_ij

    return mat_r


def D_rmsd(i, j, mat_r):
    # calculate RMSD of distance matrix for two structures
    D_sum = 0
    for k in range(mat_r.shape[1]):
        d_ij = abs(mat_r[i][k] - mat_r[j][k])
        D_sum += d_ij**2
    D = np.sqrt(D_sum / mat_r.shape[1])
    return D


def zwanzig(mol1, mol2, CV_list, n_bins, output_dir):
    RT = 300 * 8.3145 / 1000 / 4.184 # kcal/mol
    bin_width = 360 / n_bins
    pop = np.zeros(shape=(n_bins, n_bins))
    bin_tot = np.zeros(shape=(n_bins, n_bins))
    correction = np.zeros(shape=(n_bins, n_bins))
    for i in range(len(mol1.energies)):
        bin = np.empty(shape=[CV_list.shape[0]], dtype=int)
        rel_qm_energy = mol2.energies[i] - np.min(mol2.energies)
        rel_mm_energy = mol1.energies[i] - np.min(mol1.energies)
        energy_diff = rel_qm_energy - rel_mm_energy
        boltzmann = math.exp(-1.0*energy_diff/RT)
        for i_dih in range(CV_list.shape[0]):
            p = np.zeros([CV_list.shape[1], 3])
            p[0:] = mol1.coords[i][CV_list[i_dih][:]]
            bin[i_dih] = int((dihedral(p) + 180) / bin_width)
            if bin[i_dih] == n_bins:
                bin[i_dih] = 0
        pop[bin[1]][bin[0]] = pop[bin[1]][bin[0]] + 1
        bin_tot[bin[1]][bin[0]] = bin_tot[bin[1]][bin[0]] + boltzmann
    for i_bin in range(n_bins):
        for j_bin in range(n_bins):
            correction[j_bin][i_bin] = -1.0*RT*math.log(bin_tot[j_bin][i_bin]/pop[j_bin][i_bin])
            if correction[np.isnan(correction)]:
                print(bin_tot[j_bin][i_bin], pop[j_bin][i_bin])
                print("Error - bin population cannot be zero.")
                print("Decrease free energy surface resolution.")
                exit()
    correction = correction - np.min(correction)
    return correction