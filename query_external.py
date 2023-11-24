# import modules
import numpy as np
import calc_geom, output
import matplotlib.pyplot as plt

def geom(sample_freq, molecule, source, output_dir):

    # load dataset and assign to arrays - will need to change this on CSF
    data_set = np.load(f'/Users/user/datasets/{source}/{source}_{molecule}.npz')

    atom_indices = input("""
        Enter atom indices separated by spaces:
            e.g. for a distance "0 1"
            e.g. for an angle "2 3 4"
            e.g. for a dihedral "5 4 6 10"
            Consult mapping.dat for connectivity.
        > """)

    if len(atom_indices.split()) == 2:
        print("Calculating distances...")
    elif len(atom_indices.split()) == 3:
        print("Calculating angles...")
    elif len(atom_indices.split()) == 4:
        print("Calculating dihedrals...")
    else:
        print("Invalid number of atoms")
        exit()

    # slight differences in formatting of md17/rmd17
    if source == "md17":
        positions = data_set['R']  # Angstrom
        scatter_label = "simulation time (ps)"  # sample set is a time series
        time_step = 0.0005  # ps
    elif source == "rmd17":
        positions = data_set['coords']
        scatter_label = "sample number"  # sample set is not a time series

    # length of atoms array will determine type of geomteric calculation to do
    atoms = np.array(atom_indices.split(), dtype=int)
    p = np.zeros([len(atoms), 3])
    num_data = int(positions.shape[0] / sample_freq)
    time = np.zeros(num_data)
    geom_data = np.zeros(num_data)
    for data in range(num_data):
        time[data] = data * sample_freq
        if source == "md17":
            time[data] = time[data] * time_step
        p[:] = positions[data * sample_freq][atoms[:]]
        if len(atoms) == 2:  # pair distance
            x_label = "distance (A)"
            geom_data[data] = calc_geom.distance(p)
        elif len(atoms) == 3:  # triplet angle
            x_label = "angle (degrees)"
            geom_data[data] = calc_geom.angle(p)
        elif len(atoms) == 4: # dihedral angle
            x_label = "dihedral (degrees)"
            geom_data[data] = calc_geom.dihedral(p)

    # plot distribution, scatter and save data
    hist, bin = np.histogram(geom_data, 50, (min(geom_data), max(geom_data)))
    bin = bin[range(1, bin.shape[0])]
    output.lineplot(bin, hist / num_data, "linear", x_label,
        "relative probability", "ext_geom_dist", output_dir)
    np.savetxt(f"./{output_dir}/ext_geom.dat",
        np.column_stack((bin, hist / num_data)), delimiter=" ", fmt="%.6f")
    output.scatterplot(time, geom_data, "linear", scatter_label, x_label,
        "ext_geom_sct", output_dir)
    return None


def pop2D(sample_freq, n_bins, CV_list, molecule, source, output_dir):
    data_set = np.load(f'/Users/user/datasets/{source}/{source}_{molecule}.npz')
    if source == "md17":
        coords = data_set['R']  # Angstrom
    elif source == "rmd17":
        coords = data_set['coords']
    set_size = int(coords.shape[0] / sample_freq)
    bin_width = 360 / n_bins
    pop = np.zeros(shape=(n_bins, n_bins))
    for item in range(set_size):
        bin = np.empty(shape=[CV_list.shape[0]], dtype=int)
        for i_dih in range(CV_list.shape[0]):
            p = np.zeros([CV_list.shape[1], 3])
            p[0:] = coords[item][CV_list[i_dih][:]]
            bin[i_dih] = int((calc_geom.dihedral(p) + 180) / bin_width)
        if len(bin) == 1:
            pop[bin[0]] += 1
        elif len(bin) == 2:
            pop[bin[1]][bin[0]] += 1
    pop = pop / (set_size)
    x, y = np.meshgrid(np.linspace(-180, 180, n_bins),
                       np.linspace(-180, 180, n_bins))
    output.heatmap2D(x, y, pop, pop.max(), output_dir, "pop_2d", "gist_heat", fe_map=False)
    count = 0
    for i in range(n_bins):
        for j in range(n_bins):
            if pop[i][j] != 0:
                count += 1
    print("% of surface populated:", 100*count /(n_bins*n_bins))
    return None

