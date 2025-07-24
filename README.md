This repository contains Python scripts used to simulate the evolution of a particle beam in predefined external forces by numerically solving each particle's equation of motion.

To get good statistics of beam parameters, we need a lot of particles in the simulation. Using [mpi4py](https://mpi4py.readthedocs.io/en/stable/), we parallelize the computation across different cores, decreasing the wall time of the simulation.

Some simulation results have been published in [Physics of Plasmas](https://pubs.aip.org/aip/pop/article/31/6/063106/3299178/Emittance-preservation-for-the-electron-arm-in-a).
