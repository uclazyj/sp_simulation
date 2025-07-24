import numpy as np
from scipy.integrate import solve_ivp
from scipy import interpolate
import time
from mpi4py import MPI
import json

from utils import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

### load input data from sp_input.json

input_file = load_json_file('sp_input.json')

dimension = input_file['simulation']['dimension']
transverse_motion_model = input_file['simulation']['transverse_motion_model']

s = input_file['plasma']['s']
fs = input_file['plasma']['fs']

x_init_global = input_file['beam']['x']
px_init_global = input_file['beam']['px']
gamma_init_global = input_file['beam']['gamma']
if dimension == 2:
    y_init_global = input_file['beam']['y']
    py_init_global = input_file['beam']['py']
    
s_acc_model = input_file['acc_model']['s']
gamma_gain = input_file['acc_model']['gamma_gain']

### Ion collapse model ###

ion_model = input_file['ion_model']['type']
s_ion_model = input_file['ion_model']['s']
A = input_file['ion_model']['A']
sig_i = input_file['ion_model']['sig_i']

### Focusing field model ###

s_ff = np.array(input_file['focusing_field_model']['s'])
r_ff = np.array(input_file['focusing_field_model']['r'])
ff_ff = np.array(input_file['focusing_field_model']['ff'])


delta_s_dump = input_file['diag']['delta_s_dump']

### Assign particles to different cores ###

N_global = len(x_init_global)

start_idx,end_idx = get_start_and_end_index(N_global,nprocs,rank)
N_local = end_idx - start_idx
x_init_local = x_init_global[start_idx:end_idx]
px_init_local = px_init_global[start_idx:end_idx]
gamma_init_local = gamma_init_global[start_idx:end_idx]


if dimension == 2:
    y_init_local = y_init_global[start_idx:end_idx]
    py_init_local = py_init_global[start_idx:end_idx]
    

### Ion collapse model ###

fs_model = interpolate.interp1d(s, fs)
gamma_gain_model = interpolate.interp1d(s_acc_model, gamma_gain,fill_value="extrapolate")
A_model = interpolate.interp1d(s_ion_model, A)
sig_i_model = interpolate.interp1d(s_ion_model, sig_i)

### Focusing field model ###

focusing_field_model = interpolate.interp2d(s_ff, r_ff, ff_ff.T)




### For each node, solve the equations of motion of particles (belong to this node) to get their phase space trajectories

L_total = s[-1]
s_eval = np.arange(0,L_total+1e-10, delta_s_dump)

x_local = np.zeros((N_local,len(s_eval)))
px_local = np.zeros((N_local,len(s_eval)))
if dimension == 2:
    y_local = np.zeros((N_local,len(s_eval)))
    py_local = np.zeros((N_local,len(s_eval)))

t0 = time.time()

for i in range(N_local):
    if dimension == 1:
        if transverse_motion_model == "ion":
            sol = solve_ivp(derivative,[0,L_total], [x_init_local[i],px_init_local[i]], t_eval=s_eval, \
                     args=(gamma_init_local[i],gamma_gain_model,A_model,sig_i_model,fs_model,ion_model),rtol=1e-6, atol = 1e-9)
        elif transverse_motion_model == "focusing_field":
            sol = solve_ivp(derivative2,[0,L_total], [x_init_local[i],px_init_local[i]], t_eval=s_eval, \
                     args=(gamma_init_local[i],gamma_gain_model,focusing_field_model),rtol=1e-6, atol = 1e-9)
        else:
            print('Wrong value for transverse_motion_model!')
    else:
        if transverse_motion_model == "ion":
            sol = solve_ivp(derivative,[0,L_total], [x_init_local[i],px_init_local[i],y_init_local[i],py_init_local[i]], t_eval=s_eval, \
                     args=(gamma_init_local[i],gamma_gain_model,A_model,sig_i_model,fs_model,ion_model),rtol=1e-6, atol = 1e-9)
        elif transverse_motion_model == "focusing_field":
            sol = solve_ivp(derivative2,[0,L_total], [x_init_local[i],px_init_local[i],y_init_local[i],py_init_local[i]], t_eval=s_eval, \
                     args=(gamma_init_local[i],gamma_gain_model,focusing_field_model),rtol=1e-6, atol = 1e-9)
        else:
            print('Wrong value for transverse_motion_model!')
    assert(all(sol.t==s_eval))
    x_local[i,:] = sol.y[0]
    px_local[i,:] = sol.y[1]
    if dimension == 2:
        y_local[i,:] = sol.y[2]
        py_local[i,:] = sol.y[3]

t1 = time.time()

if rank == 0:
    print('It takes '+str(round(t1-t0))+' s to calculate all the particles trajectories in phase space!')

### Gather all the computation results in master node 0
    
x_locals_array = comm.gather(x_local, root=0)
px_locals_array = comm.gather(px_local, root=0)

if dimension == 2:
    y_locals_array = comm.gather(y_local, root=0)
    py_locals_array = comm.gather(py_local, root=0)
    
### Save results to file: sp_output.json 
    
if rank == 0:
    output_dic = {}
    # convert an array of 2d arrays into one large 2d array
    x_global = np.vstack(x_locals_array)
    px_global = np.vstack(px_locals_array)
    
    output_dic['x'] = x_global.tolist()
    output_dic['px'] = px_global.tolist()
    if dimension == 2:
        y_global = np.vstack(y_locals_array)
        py_global = np.vstack(py_locals_array)
        
        output_dic['y'] = y_global.tolist()
        output_dic['py'] = py_global.tolist()

    # print('x_global.shape = ',x_global.shape)
    
    fs_eval = fs_model(s_eval)
    gamma_gain_eval = gamma_gain_model(s_eval)
    A_eval = A_model(s_eval)
    sig_i_eval = sig_i_model(s_eval)
    output_dic['s'] = s_eval.tolist()
    output_dic['fs'] = fs_eval.tolist()
    output_dic['gamma_gain'] = gamma_gain_eval.tolist()
    output_dic['A'] = A_eval.tolist()
    output_dic['sig_i'] = sig_i_eval.tolist()
    
    t_i = time.time()
    
    save_to_json_file(output_dic,'sp_output.json')
    
    t_f = time.time()
    print('It takes '+str(round(t_f-t_i))+' s to save all the particle data!')