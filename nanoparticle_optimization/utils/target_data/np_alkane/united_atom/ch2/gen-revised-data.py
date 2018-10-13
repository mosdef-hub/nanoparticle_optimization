import numpy as np

#radii = np.arange(2, 11)
radii = [9]

for radius in radii:
    # Load the initial target data obtained
    initial_data = np.loadtxt('U_{:.0f}nm_CH2_init.txt'.format(radius))

    # Load the additional data obtained around the well region
    all_data = np.loadtxt('U_{:.0f}nm_CH2_well.txt'.format(radius))
    well_data = []
    for point in all_data:
        if(point[1] not in initial_data[:,1] and \
                point[2] not in initial_data[:,2]):
            well_data.append(point)
    well_data = np.array(well_data)
    well_data = well_data[::2]

    revised_initial_data = []
    for point in initial_data:
        if point not in well_data:
            revised_initial_data.append(point)
    revised_initial_data = np.array(revised_initial_data)

    total = np.vstack((revised_initial_data, well_data))
    total = total[np.argsort(total[:, 0])]

    # Include only monotonically decreasing points up to the bottom
    # of the well, and only monotonically increasing points after
    # the well
    argmin_U = np.argmin(total[:, 1])
    after_well = total[:argmin_U]
    before_well = total[argmin_U:]
    before_well_monotonic = [before_well[0]]
    after_well_monotonic = [after_well[-1]]
    for i, point in enumerate(before_well[1:]):
        if point[1] < before_well_monotonic[-1][1]:
            continue
        before_well_monotonic.append(point)

    for i, point in enumerate(after_well[-2::-1]):
        if point[1] < after_well_monotonic[-1][1]:
            continue
        after_well_monotonic.append(point)
    after_well_monotonic.reverse()

    threshold = -0.2
    after_well_monotonic = np.array([val for val in after_well_monotonic
                                     if val[1] <= threshold])
    before_well_monotonic = np.array(before_well_monotonic)

    total = np.vstack((after_well_monotonic, before_well_monotonic))

    np.savetxt('U_{:.0f}nm_CH2_revised.txt'.format(radius),
               total[np.argsort(total[:,0])])
