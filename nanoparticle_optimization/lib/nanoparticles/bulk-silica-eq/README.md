The scripts and files contains within this directory pertain to the equilbration of
the amorphous silica bulk from which atomistic nanoparticles are carved. The contents
of this directory are as follows:

* data.silica\_box.txt - The initial box, a stoichiometric ratio of Si and O at
  a density of 2.2 g/mL.
* data.SiO2\_quench\_10ps\_reax.txt - The final box. This is converted to PDB
  format for nanoparticle carving.
* ffield.reax - Contains the ReaxFF parameters for silica, of Fogarty et al, used
  in box equilibration.
* in.SiO2\_melt\_reax.txt - LAMMPS input script, melt the initial structure at
  a temperature of 10000K.
* in.SiO2\_quench\_10ps\_reax.txt - LAMMPS input script, quench the melted silica
  to form an amorphous solid following the stepwise procedure of Litton and
  Garolfalini
