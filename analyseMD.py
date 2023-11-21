import numpy as np
import calc_geom
import output
from scipy.stats import binned_statistic

def fes2D(input_dir, output_dir):
    x_count = 0
    y_count = 0
    with open(f"{input_dir}/fes.dat", "r") as input:
        for line in input:
            if line.strip():
                if line.startswith("#"):
                    if "nbins_phi" in line:
                        n_bins_x = int(line.strip('\n').split()[-1])
                    elif "nbins_psi" in line:
                        n_bins_y = int(line.strip('\n').split()[-1])
                        FE = np.zeros(shape=(n_bins_x, n_bins_y))
                    continue
                FE[x_count, y_count] = float(line.strip('\n').split()[2])/4.184
                y_count += 1
                if y_count == n_bins_y:
                    x_count += 1
                    y_count = 0
                if x_count == n_bins_x:
                    break
    input.close()
    x, y = np.meshgrid(np.linspace(-180, 180, n_bins_x),
                       np.linspace(-180, 180, n_bins_y))
    output.heatmap2D(x, y, FE, FE.max(), output_dir, "fes_2d", "RdBu",fe_map=True)
    return None

def pop2D(mol1, n_bins, CV_list, output_dir, init, set_size):
    bin_width = 360 / n_bins
    pop = np.zeros(shape=(n_bins, n_bins))
    for item in range(init, set_size):
        bin = np.empty(shape=[CV_list.shape[0]], dtype=int)
        for i_dih in range(CV_list.shape[0]):
            p = np.zeros([CV_list.shape[1], 3])
            p[0:] = mol1.coords[item][CV_list[i_dih][:]]
            bin[i_dih] = int((calc_geom.dihedral(p) + 180) / bin_width)
        if len(bin) == 1:
            pop[bin[0]] += 1
        elif len(bin) == 2:
            pop[bin[1]][bin[0]] += 1
    pop = pop / (set_size - init)
    x, y = np.meshgrid(np.linspace(-180, 180, n_bins),
                       np.linspace(-180, 180, n_bins))
    output.heatmap2D(x, y, pop, pop.max(), output_dir, "pop_2d", "gist_heat",fe_map=False)
    return None


def pop1D(mol1, n_bins, CV_list, output_dir, init, set_size):
    dih = np.zeros(shape=(set_size-init))
    for item in range(init, set_size):
        p = np.zeros([CV_list.shape[1], 3])
        p[0:] = mol1.coords[item][CV_list[0][:]]
        dih[item-init] = calc_geom.dihedral(p)
    hist, bin = np.histogram(dih, n_bins, (-180, 180))
    bin = bin[range(1, bin.shape[0])]
    output.lineplot(bin, hist / (set_size - init), "linear", "pop_1d",
    "probability", "pop_1d", output_dir)
    np.savetxt(f"./{output_dir}/pop_1d.dat", np.column_stack((bin,
        hist / (set_size - init))), delimiter=" ", fmt="%.6f")


def force_MSE_dist(baseline, values, output_dir):
    RSE = np.sqrt((baseline - values)**2)
    MSE = binned_statistic(baseline, RSE, statistic='mean',
        bins=50, range=(-200.0, 200.0))
    bin_width = (MSE.bin_edges[1] - MSE.bin_edges[0])
    bin_centers = MSE.bin_edges[1:] - bin_width / 2
    output.lineplot(bin_centers, MSE.statistic, "linear",
        "QM force (kcal/mol/$\AA$)", "MSE (kcal/mol/$\AA$)", "force_MSE_dist", output_dir)
    np.savetxt(f"./{output_dir}/force_mse_dist.dat", np.column_stack((bin_centers,
        MSE.statistic)), delimiter = " ", fmt="%.6f")
    output.scatterplot(baseline, RSE, "linear", "QM force (kcal/mol/$\AA$)",
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
    output.lineplot(bin, hist / bin_width / set_size, "linear", "energy",
        "probability", "mm_energy_dist", output_dir)
    np.savetxt(f"./{output_dir}/mm_energy_dist.dat", np.column_stack((bin,
        hist / bin_width / set_size)), delimiter=" ", fmt="%.6f")
    output.scatterplot(rel_baseline, rel_values, "linear", "Rel. QM energy (kcal/mol)",
        "Rel. MD energy (kcal/mol)", "energy_error_scatter", output_dir)
    return None


def check_stability(mol1, init, set_size, output_dir):
    # calculate bond distance list for equilibrium structure
    n_atoms = len(mol1.atoms)
    max_r = 1.6
    max_bonds = n_atoms * 3
    atom_indices = np.zeros([max_bonds, 2], dtype=int)
    bond_dist = np.zeros([max_bonds], dtype=float)
    n_bonds = 0
    for i in range(n_atoms):
        for j in range(i):
            r_ij = np.linalg.norm(mol1.coords[init][i] - mol1.coords[init][j])
            if r_ij < max_r:
                atom_indices[n_bonds][0] = i
                atom_indices[n_bonds][1] = j
                bond_dist[n_bonds] = r_ij
                n_bonds += 1

    # check stability of trajectory
    stable = True

    print("Checking bond distance deviation criteria...")
    max_dev = 0.25 # maximum bond distance deviation
    element = {1: "H", 6: "C", 7: "N", 8: "O"}
    atom_names = []
    for i in range(len(mol1.atoms)):
        atom_names.append(element[mol1.atoms[i]])
    for s in range(init, set_size):
        for i_bond in range(n_bonds):
            p = np.zeros([2, 3])
            p[:] = mol1.coords[s][atom_indices[i_bond][:]]
            r_ij = calc_geom.distance(p)
            if abs(r_ij - bond_dist[i_bond]) > max_dev:
                print("frame, bond number, atom i, atom j, bond dist (A), ref dist (A)")
                print(s, i_bond, atom_indices[i_bond][0],
                    atom_indices[i_bond][1], r_ij, bond_dist[i_bond])
                print("Writing .pdb file...")
                output.write_pdb(mol1.coords[s][:][:], "name", 1, mol1.atoms,
                    atom_names, f"./{output_dir}/mol_{s + 1}.pdb", "w")
                stable = False

    # check that there are no close contacts
    print("Checking close contacts...")
    min_dist = 0.75 # minimum interatomic distance
    for s in range(init, set_size):
        for i in range(n_atoms):
            for j in range(i):
                r_ij = np.linalg.norm(mol1.coords[s][i] - mol1.coords[s][j])
                if r_ij < min_dist:
                    print("frame, bond number, atom i, atom j, bond dist (A), ref dist (A)")
                    print(s, i_bond, atom_indices[i_bond][0],
                          atom_indices[i_bond][1], r_ij, bond_dist[i_bond])
                    print("Writing .pdb file...")
                    output.write_pdb(mol1.coords[s][:][:], "name", 1, mol1.atoms,
                        atom_names, f"./{output_dir}/mol_{s + 1}.pdb", "w")
                    stable = False

    if stable:
        print("Trajectory is stable - no unphysical structures generated.")
    else:
        print("Trajectory is unstable - unphysical structures generated.")

    return None