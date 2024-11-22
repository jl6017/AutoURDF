#!/bin/bash
## default setting, 3 cameras
python Sim/sim_data.py --robot wx200_5 --ground
python Sim/sim_data.py --robot franka 
python Sim/sim_data.py --robot ur5 

python Sim/sim_data.py --robot bolt
python Sim/sim_data.py --robot solo8

python Sim/sim_data.py --robot pxs
python Sim/sim_data.py --robot allegro_12
python Sim/sim_data.py --robot op3

