#!/bin/bash

module unload desimodules
source /project/projectdirs/desi/software/desi_environment.sh 18.7

PREFIX=/global/homes/q/qmxp55/DESI/matches

cd $PREFIX

python generate_cat.py
