#!/bin/bash

#Please, input file should respect the next format:
#N_electrons  =  2
#N_phonons    =  1
#N_phon_levels =  20
#N_bath       =  5000
#Elec_levels  =  0.0 0.07
#fb_vec       =  0.0 0.0 #λ = 1eV -> fb = 0.0072279/N_bath a.u.
#Phon_freq    =  0.00734987
#Phon_mass    =  3645.777
#K0_bath      =  0.0 #1eV/A^2 = 0.010277 a.u.
#K_ceed       =  0.0
#Field_amp    =  0.0
#Total_time   =  50000
#Delta_t      =  0.02
#print_step   =  200
#bath_temp    =  0.0007
#seed         =  25091993

./program.out

echo END
