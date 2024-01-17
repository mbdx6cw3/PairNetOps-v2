import numpy as np
import calc_geom
import output
from scipy.stats import binned_statistic

def dist(mol, set_size, output_dir):
    n_atom = len(mol.atoms)
    hist, bin = np.histogram(mol.forces.flatten(),200,(-250,250))
    bin = bin[range(1, bin.shape[0])]
    bin_width = bin[1] - bin[0]
    output.lineplot(bin, hist / bin_width / set_size / 3.0 / n_atom, "linear",
                "force (kcal/mol/A)", "probability", "force_dist", output_dir)
    np.savetxt(f"./{output_dir}/force_dist.dat",
        np.column_stack((bin, hist / bin_width / set_size / 3.0 / n_atom)),
        delimiter = " ",fmt="%.6f")
    hist, bin = np.histogram(mol.energies, 50, (np.min(mol.energies),np.max(mol.energies)))
    bin = bin[range(1, bin.shape[0])]
    bin_width = bin[1] - bin[0]
    output.lineplot(bin, hist / bin_width / set_size, "linear", "energy",
        "probability", "energy_dist", output_dir)
    np.savetxt(f"./{output_dir}/energy_dist.dat",
        np.column_stack((bin, hist / bin_width / set_size)), delimiter = " ",
        fmt="%.6f")
    return None


def charge_dist(mol, index, set_size, output_dir):
    charge = np.empty(shape=[set_size])
    for item in range(set_size):
        charge[item] = mol.charges[item][index]
    hist, bin = np.histogram(charge,50,(np.min(charge),np.max(charge)))
    bin = bin[range(1, bin.shape[0])]
    bin_width = bin[1] - bin[0]
    output.lineplot(bin, hist / bin_width / set_size, "linear",
                "charge (e)", "probability", "charge_dist", output_dir)
    np.savetxt(f"./{output_dir}/charge_dist.dat",
        np.column_stack((bin, hist / bin_width / set_size)),
        delimiter = " ",fmt="%.6f")
    return None


# charge_CV here.


def energy_CV(mol, atom_indices, set_size, output_dir):
    CV_list = np.array(atom_indices.split(), dtype=int)
    CV = np.empty(shape=[set_size])
    for item in range(set_size):
        p = np.zeros([len(CV_list), 3])
        p[0:] = mol.coords[item][CV_list[:]]
        if len(CV_list) == 2:
            x_label = "$r_{ij} / \AA$"
            CV[item] = calc_geom.distance(p)
        elif len(CV_list) == 3:
            x_label = "$\u03F4_{ijk}  (degrees)$"
            CV[item] = calc_geom.angle(p)
        elif len(CV_list) == 4:
            x_label = "$\u03C6_{ijkl} (degrees)$"
            CV[item] = calc_geom.dihedral(p)
    # plot distribution, scatter and save data
    print("MEAN:", np.mean(CV))
    energy = mol.energies[:,0] - np.min(mol.energies[:,0])
    hist, bin = np.histogram(CV, 50, (min(CV), max(CV)))
    bin = bin[range(1, bin.shape[0])]
    output.lineplot(bin, hist / set_size, "linear", x_label,
        "relative probability", "geom_dist", output_dir)
    np.savetxt(f"./{output_dir}/geom_dist.dat",
        np.column_stack((bin, hist / set_size)), delimiter=" ", fmt="%.6f")
    output.scatterplot(CV, energy, "linear", x_label,
        "QM energy (kcal/mol)", "energy_geom_scatter", output_dir)
    np.savetxt(f"./{output_dir}/energy_geom_scatter.dat",
        np.column_stack((CV, energy)), delimiter=" ", fmt="%.6f")
    #means, edges, counts = binned_statistic(CV, energy, statistic='min',
    #    bins=72, range=(-180.0, 180.0))
    #bin_width = edges[1] - edges[0]
    #bin_centers = edges[1:] - bin_width / 2
    #output.lineplot(bin_centers, means, "linear", x_label,
    #    "mean energy (kcal/mol)", "qm_energy_geom", output_dir)
    #np.savetxt(f"./{output_dir}/qm_energy_geom.dat",
    #    np.column_stack((bin_centers, means)), delimiter = " ",
    #           fmt="%.6f")
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
        mol.output_eij[s] = np.matmul(np.linalg.pinv(e_ij), mol.energies[s])

    # flatten output_matFE instead below?
    output.scatterplot([mol.mat_r.flatten()], [mol.output_eij.flatten()], "linear",
        "$r_{ij}$ / $\AA$", "$e_{ij}$ / kcal/mol", "eij_rij", output_dir)
    np.savetxt(f"./{output_dir}/eij_rij.dat",
        np.column_stack((mol.mat_r.flatten(), mol.output_eij.flatten())),
        delimiter=" ", fmt="%.6f")
    hist, bin = np.histogram(mol.output_eij.flatten(), 200,
        (np.min(mol.output_eij.flatten()), np.max(mol.output_eij.flatten())))
    bin = bin[range(1, bin.shape[0])]
    bin_width = bin[1] - bin[0]
    output.lineplot(bin, hist / bin_width / _NC2 / set_size, "linear",
        "$e_{ij}$ / kcal/mol", "probability", "eij_dist", output_dir)
    np.savetxt(f"./{output_dir}/eij_dist.dat",
               np.column_stack((bin, hist / bin_width / _NC2 / set_size)),
               delimiter=" ", fmt="%.6f")
    return None


# TODO: this function will be removed.
def get_NRF(zA, zB, r):
    _NRF = r and (zA * zB * np.float64(627.5095 * 0.529177) / (r ** 2))
    return _NRF


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

