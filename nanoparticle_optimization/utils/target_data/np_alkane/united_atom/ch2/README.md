Target data for NP-alkane interactions was a bit tricky to collect as following the initial data collection it was found that additional data was needed in the region of the potential well and in the long-range region. Furthermore, the high error in the short range region did not result in a potential that monotonically decreased to the bottom of the well and monotonically increased afterward, due to slight fluctuations in the data. As a result, a function was applied to the data to ensure it satisfied that criteria.

This led to the collection of several different sets of target data that were eventually combined into the final set (the files appended with "-complete"). Rather than removing these older files, everything is included in this directory. The following provides a brief description of the five files present for each radius:

    - U_Xnm_CH2_init.txt
        The initial target data collected (corresponding C++ file is pe_np_ch2_mpi-init.cpp)
        Pruned to remove the first 10 bins
    - U_?nm_CH2_well.txt
        U_?nm_CH2_init.txt + additional data collected in the
        short-range region (corresponding C++ file is pe_np_ch2_mpi-well.cpp)
    - U_?nm_CH2-lr.txt
        Data collected in the region of long range (corresponding C++ file is pe_np_ch2_mpi-lr.cpp)
    - U_?nm_CH2-revised.txt
        Combination of "-init" and "-well" data ensuring that points monotonically decrease to the bottom of the potential well and monotonically increase after.
    - U_?nm_CH2-complete.txt
        Combination of "-init", "-well", and "-lr" data ensuring that points monotonically decrease to the bottom of the potential well and monotonically increase after.
