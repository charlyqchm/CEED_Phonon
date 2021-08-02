#!/bin/bash

#Please, input file should respect the next format:
#N_electrons:    2
#N_phonons:      2
#N_phon_levels:  4
#N_bath:        10
#Elec_levels:  0.0 1.0
#fb_vec:       0.0 1.0
#Phon_freq:    0.1 0.15
#Phon_mass:    1.0 1.0
#K0_bath:      1.0
#K_ceed:       1.0
#Field_amp:    1.0
#Total_time:   1000
#Delta_t:      0.1
#print_step:   10

input=$1

awk '{$1=""}1' "$input" > input.in

./program.out

echo END
