import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

plt.rcParams["font.size"] = 20
#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'custom'
#plt.rcParams['mathtext.it'] = 'Times New Roman' #italic?
#plt.rcParams['mathtext.rm'] = 'Times New Roman'

def lineplot(x, y, type, x_label, y_label, title, output_dir):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.xscale(type)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"./{output_dir}/{title}.png")
    return None

def twolineplot(x1, x2, y1, y2, l1, l2, type, x_label, y_label, title,
                output_dir):
    fig, ax = plt.subplots()
    ax.plot(x1, y1, label = l1)
    ax.plot(x2, y2, label = l2)
    plt.xscale(type)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"./{output_dir}/{title}.png")
    return None

def scatterplot(x, y, type, x_label, y_label, title, output_dir):
    fig, ax = plt.subplots()
    ax.scatter(x, y, marker="o", c="black", s=1)
    plt.xscale(type)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"./{output_dir}/{title}.png")
    return None

def grotrj(n_atom, res, vecs, time, coords, vels, atom_names, output_dir, file_name):

    if time == 0.0:
        open_type = "w"
    else:
        open_type = "a"
    gro_file = open(f"{output_dir}/{file_name}.gro", open_type)
    gro_file.write(f"output t={time}\n")
    gro_file.write(f"{n_atom}\n")
    count = -1
    for i_res in range(len(res)):
        for i_atm in range(len(list(res[i_res].atoms()))):
            count += 1
            gro_file.write(f"{i_res+1:>5}{res[i_res].name:3} "
                f"{atom_names[count]:>6} {count:>4} "
                f"{coords[count][0]:7.3f} {coords[count][1]:7.3f} "
                f"{coords[count][2]:7.3f} {vels[count][0]:7.3f} "
                f"{vels[count][1]:7.3f} {vels[count][2]:7.3f}\n")
    gro_file.write(f"{vecs[0]:10.5f}{vecs[0]:10.5f}{vecs[0]:10.5f}\n")
    gro_file.close()
    return None


def gro(mol, output_dir):
    vecs = [2.5,2.5,2.5]
    mol.coords = mol.coords / 10 # convert to nm
    for item in range(mol.coords.shape[0]):
        gro_file = open(f"./{output_dir}/mol_{item + 1}.gro", "w")
        gro_file.write(f"output t=0.0 ps\n")
        gro_file.write(f"{mol.n_atom}\n")
        for i_atm in range(mol.n_atom):
            gro_file.write(f"    1MOL"
                f"{mol.atom_names[i_atm]:>6} {i_atm:>4} "
                f"{mol.coords[item][i_atm][0]:7.3f} "
                f"{mol.coords[item][i_atm][1]:7.3f} "
                f"{mol.coords[item][i_atm][2]:7.3f}\n")
        gro_file.write(f"{vecs[0]:10.5f}{vecs[0]:10.5f}{vecs[0]:10.5f}\n")
        gro_file.close()
    return None


def pdb(mol, output_dir, s):
    '''
    Write 3D coords in .pdb format. Provide the resname, resid, atoms
    and filename.
    '''
    if s == "none":
        for s in range(len(mol.coords)):
            print(s)
            file = open(f"./{output_dir}/mol_{s+1}.pdb", "w")
            file.write("COMPND    UNNAMED\n")
            file.write("Generated by PairNet\n")
            for i in range(mol.n_atom):
                file.write(f"ATOM{i+1:7}  {mol.atom_names[i]:4}MOL     1 "
                    f"{mol.coords[s][i][0]:11.3f} {mol.coords[s][i][1]:7.3f} "
                    f"{mol.coords[s][i][2]:7.3f} 1.0 0.0 {mol.atom_names[i]:4}\n")
            file.write('TER\n')
            file.close()
    else:
        file = open(f"./{output_dir}/mol_{s+1}.pdb", "w")
        file.write("COMPND    UNNAMED\n")
        file.write("Generated by PairNet\n")
        for i in range(mol.n_atom):
            file.write(
                f"ATOM{i + 1:7}  {mol.atom_names[i]:4}MOL     1 "
                f"{mol.coords[s][i][0]:11.3f} {mol.coords[s][i][1]:7.3f} "
                f"{mol.coords[s][i][2]:7.3f} 1.0 0.0 {mol.atom_names[i]:4}\n")
        file.write('TER\n')
        file.close()
    return None


def xyz(mol, output_dir):
    '''
    For a given set of structure 3D coords and the atom
    associated numbers, output xyz format.
    '''
    xyz_file = open(f"./{output_dir}/data.xyz", "w")

    for item in range(len(mol.energies)):
        xyz_file.write(f"{mol.coords.shape[1]}\n")
        xyz_file.write(f"\"Lattice=50.0 0.0 0.0 0.0 50.0 0.0 0.0 0.0 50.0\" "
                       f"Properties=species:S:1:pos:R:3:forces:R:3 "
                       f"energy={mol.energies[item]} pbc=\"F F F\"\n")
        for atom in range(mol.coords.shape[1]):
            r = mol.coords[item][atom]
            f = mol.forces[item][atom]
            xyz_file.write('{:4} {:11.6f} {:11.6f} {:11.6f} {:11.6} {:11.6} {:11.6}\n'.format(
                    mol.atom_names[atom], r[0], r[1], r[2], f[0], f[1], f[2]))
    xyz_file.close()
    return None


def scurve(baseline, values, output_dir, output_file, val):
    """
    This function calculates S-curves for MM determined forces.
    Must have run qm2ml.py first to collect QM force dataset.
    :param molecule:
    :param set_size:
    :param n_atom:
    :return:
    """
    RSE = np.sqrt((baseline-values)**2)
    hist, bin_edges = np.histogram(RSE,1000,(-0.2,np.amax(RSE)))
    hist = np.cumsum(hist)
    bin_edges = bin_edges[range(1,bin_edges.shape[0])]
    hist = hist/values.shape[0]*100
    bin_edges[0] = 0.0
    lineplot(bin_edges, hist, "log", "Error", "% of forces below error",
             f"{output_file}", output_dir)
    np.savetxt(f"./{output_dir}/{output_file}.dat", np.column_stack((bin_edges,
        hist)), fmt='%.6f', delimiter = " ")
    i_L = (np.abs(bin_edges - val)).argmin()
    return hist[i_L]


def heatmap2D(x, y, z, output_dir, file, cmap, map_type):
    fig, ax = plt.subplots()
    if map_type == 0:       # free energy map
        c = ax.pcolormesh(x, y, z, cmap=cmap)
    elif map_type == 1:     # population map
        z_max = z.max()
        c = ax.pcolormesh(x, y, z, norm=colors.LogNorm(vmin=0.00001,vmax=z_max),
             cmap=cmap)
    elif map_type == 2:     # force error map
        c = ax.pcolormesh(x, y, z, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    x_label = "$\u03C6$ ($\u00b0$)"
    y_label = "$\u03C8$ ($\u00b0$)"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    fig.colorbar(c, ax=ax)
    plt.savefig(f"./{output_dir}/{file}.png", bbox_inches="tight")
    return None


def gau(mol, coords, output_dir, opt, CV_list):

    # if optimisations requested we also need to read in opt text sections
    gaussian = open(f"./gaussian.txt", "r")
    text = gaussian.read().strip('\n')

    if "charge" in text:
        charges = np.reshape(np.loadtxt("md_data/background_charges.txt",
            dtype=float), (len(coords), -1, 4))

    # if doing constrained optimisation read in torsional CVs
    if opt:
        CV_list = [i + 1 for i in CV_list]
        CV_text = " \n" + " ".join(str(i) for i in CV_list) + " B \n" \
            + " ".join(str(i) for i in CV_list) + " F \n"

    # create QM input files
    for item in range(len(coords)):
        new_text = text.replace("index", f"{item+1}")
        coord_text = ""
        charge_text = ""
        for atom in range(mol.n_atom):
            coord_atom = f"{mol.atom_names[atom]} " \
                         f"{coords[item,atom,0]:.8f} " \
                         f"{coords[item,atom,1]:.8f} " \
                         f"{coords[item,atom,2]:.8f} \n"
            coord_text = coord_text + coord_atom
        if "charge" in text:
            coord_text = coord_text + "\n"
            for atom in range(charges.shape[1]):
                charge_atom = f"{charges[item,atom,0]:.8f} " \
                         f"{charges[item,atom,1]:.8f} " \
                         f"{charges[item,atom,2]:.8f} " \
                         f"{charges[item,atom,3]:.8f} \n"
                charge_text = charge_text + charge_atom
            coord_text = coord_text + charge_text

        if opt:
            coord_text = coord_text + CV_text

        new_text = new_text.replace("COORDS", coord_text)
        qm_file = open(f"./{output_dir}/mol_{item+1}.gjf", "w")
        print(new_text, file=qm_file)
        print(file=qm_file)
        qm_file.close()
    return None


def dataset(mol, output_dir):
    print("Writing output...")
    # write .txt files
    coord_file = open(f"./{output_dir}/coords.txt", "w")
    energy_file = open(f"./{output_dir}/energies.txt", "w")
    force_file = open(f"./{output_dir}/forces.txt", "w")
    error_file = open(f"./{output_dir}/errors.txt", "w")
    charge_file = open(f"./{output_dir}/charges.txt", "w")

    for i_file in range(len(mol.energies)):
        print(mol.energies[i_file], file=energy_file)

        # save coordinates and forces (converting to kcal/mol/A)
        for i_atom in range(mol.n_atom):
            print(*mol.coords[i_file, i_atom], file=coord_file)
            print(*mol.forces[i_file, i_atom], file=force_file)
            print(mol.charges[i_file, i_atom], file=charge_file)

    coord_file.close()
    energy_file.close()
    force_file.close()
    charge_file.close()
    error_file.close()

    return None


def violin(force_ref, force_pred, energy_ref, energy_pred, charge_ref, charge_pred,
           output_dir, file):
    f_rse = force_pred - force_ref
    e_rse = energy_pred - energy_ref
    q_rse = charge_pred - charge_ref
    rse = [e_rse, f_rse, q_rse]
    fig, ax = plt.subplots()
    extrema = False
    means = False
    colors = ["silver", "wheat", "lightblue"]
    edge_color = "black"
    labels = ["forces", "energies", "charges"]
    plot = ax.violinplot(dataset=rse, showextrema=extrema, showmeans=means)
    ax.set_ylim((-1, 1))
    ax.set_xticks([y + 1 for y in range(len(rse))], labels=labels)
    ax.tick_params(bottom=False)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    for pc, color in zip(plot['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor(edge_color)
        pc.set_alpha(1)
    plt.savefig(f"./{output_dir}/{file}.png", bbox_inches="tight")
    return