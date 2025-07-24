import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import h5py

### Input and Output

def load_json_file(path):
    with open(path) as f: 
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    return inputDeck

def save_to_json_file(inputDeck,path):
    with open(path,'w') as outfile:
        json.dump(inputDeck,outfile,indent=4)
        
"""
# items_to_save is a list of tuple. For each tuple, it is a (name,array) pair.
def save_particles_data(filename,items_to_save):
    dic = {name: value.tolist() if type(value) == np.ndarray else value for name,value in items_to_save}
    with open(filename,'w') as f:
        json.dump(dic,f,indent=4)
        
def load_particles_data(filename):
    with open(filename) as f:
        parameters = json.load(f)
    parameters = {k:np.array(parameters[k]) if type(parameters[k]) == list else parameters[k] for k in parameters}
    return parameters
"""

### Initialization

### Different plasma density ramps ###
def linear_upramp(z,L_ramp,n_entrance_over_n0):
    k = (1 - n_entrance_over_n0) / L_ramp
    return k * z + n_entrance_over_n0

def smooth_upramp(beta_m0,n_entrance_over_n0,L_ramp=None,alpha_mi = None): 
    if L_ramp == None and alpha_mi == None:
        print('Need either L_ramp or alpha_mi!')
        return
    if L_ramp != None:
        if alpha_mi != None:
            print('Warning! Redundant parameters! Only L_ramp is used, alpha_mi is not used!')
        alpha_mi = beta_m0 / L_ramp * (1 / np.sqrt(n_entrance_over_n0) - 1)    
    else:
        L_ramp = beta_m0 / alpha_mi * (1 / np.sqrt(n_entrance_over_n0) - 1)
    s = np.linspace(0,L_ramp,int(L_ramp)+1)
    fs = 1 / (1 + alpha_mi * (s-L_ramp) ** 2 / beta_m0 / L_ramp)**2
    return (s,fs,alpha_mi)

def new_smooth_upramp(beta_m0,n_entrance_over_n0,L_ramp=None,A=None): # A = the max(alpha_m)
    if L_ramp == None and A == None:
        print('Need either L_ramp or A (the upper limit of alpha_m)!')
        return
    if L_ramp != None:
        if A != None:
            print('Warning! Redundant parameters! Only L_ramp is used, A (the upper limit of alpha_m) is not used!')
        A = (1 / np.sqrt(n_entrance_over_n0) - 1) * np.pi * beta_m0 / 4 / L_ramp
    else:
        L_ramp = (1 / np.sqrt(n_entrance_over_n0) - 1) * np.pi * beta_m0 / 4 / A
    s = np.linspace(0,L_ramp,int(L_ramp)+1)
    fs = 1 / (1 + 2 * A * L_ramp / np.pi / beta_m0 * (1 + np.cos(np.pi * s / L_ramp))) ** 2
    return s,fs,A

def xinlu_upramp(beta_m0,n_entrance_over_n0,L_ramp=None,alpha_mi = None): 
    if L_ramp == None and alpha_mi == None:
        print('Need either L_ramp or alpha_mi!')
        return
    if L_ramp != None:
        if alpha_mi != None:
            print('Warning! Redundant parameters! Only L_ramp is used, alpha_mi is not used!')
        alpha_mi = beta_m0 / 2 / L_ramp * (1 / np.sqrt(n_entrance_over_n0) - 1)    
    else:
        L_ramp = beta_m0 / 2 / alpha_mi * (1 / np.sqrt(n_entrance_over_n0) - 1)
    s = np.linspace(0,L_ramp,int(L_ramp)+1)
    fs = 1 / (1 - 2 * alpha_mi * (s-L_ramp) / beta_m0)**2
    return (s,fs,alpha_mi)

def robert_upramp(L_ramp,n_entrance_over_n0):
    a = L_ramp / np.sqrt(1 / n_entrance_over_n0 - 1)
    s = np.linspace(0,L_ramp,int(L_ramp)+1)
    fs = 1 / (1 + (s - L_ramp) ** 2 / a ** 2 )
    return s,fs,a

def warren_ramp(L_ramp,n_entrance_over_n0 = 0):
    s = np.linspace(0,L_ramp,int(L_ramp)+1)
    ss = s / L_ramp
    fs = 6 * ss ** 5 - 15 * ss ** 4 + 10 * ss ** 3
    fs = (1 - n_entrance_over_n0) * fs + n_entrance_over_n0
    return s,fs

# def new_smooth_upramp(z,beta_m0,L_ramp,n_entrance_over_n0):
#     A = (1 / np.sqrt(n_entrance_over_n0) - 1) * np.pi * beta_m0 / 4 / L_ramp
#     fz = 1 / (1 + 2 * A * L_ramp / np.pi / beta_m0 * (1 + np.cos(np.pi * z / L_ramp))) ** 2
#     return fz

# def robert_upramp(z,L_ramp,n_entrance_over_n0): 
#     a = L_ramp / np.sqrt(1 / n_entrance_over_n0 - 1)
#     zz = L_ramp - z
#     return 1 / (1 + (zz/a)**2)

def save_s_fs(s,fs,path = 'sp_input.json'):
    with open(path) as f:
        input_file = json.load(f,object_pairs_hook=OrderedDict)
    input_file['plasma']['s'] = list(s)
    input_file['plasma']['fs'] = list(fs)
    with open(path,'w') as outfile:
        json.dump(input_file,outfile,indent=4)
        
        
def plot_s_fs_input(path = 'sp_input.json',dash_lines = []):
    with open(path) as f:
        input_file = json.load(f,object_pairs_hook=OrderedDict)
    s = input_file['plasma']['s']
    fs = input_file['plasma']['fs']
    plt.plot(s,fs)
    
    for i in dash_lines:
        plt.axvline(x=i,color = 'r',linestyle = '--')
    
    plt.xlabel('$z\;\;(c/\omega_{p0})$')
    plt.ylabel('$n(z)/n_0$')
    plt.title('Plasma density profile')
    plt.show()
    
### Initialize a beam with beta_i and alpha_i
### Need to propagate the Twiss parameters to the vacuum waist
### emittance is the normalized emittance
def Twiss_para_init(alpha_i,beta_i,emittance,gamma,N): 

    gamma_Twiss = (1 + alpha_i ** 2) / beta_i
    z_position = - alpha_i / gamma_Twiss

    beta_waist = 1 / gamma_Twiss
    sigma_waist = np.sqrt(beta_waist * emittance / gamma)
    sigma_p_waist = gamma * np.sqrt(gamma_Twiss * emittance / gamma)

    x = np.random.normal(0, sigma_waist, N)
    p = np.random.normal(0, sigma_p_waist, N)
    x = x + p / gamma * z_position
    alpha = -(x * p).mean() / emittance
    #print('alpha=',alpha)
    return (x,p)

def phase_space_rings_init(x_max,px_max,N):
    delta_theta = 2 * np.pi / N
    theta_all = np.arange(N) * delta_theta
    x_all = x_max * np.cos(theta_all)
    px_all = px_max * np.sin(theta_all)
    return x_all,px_all

def save_particle_coordinates(x,px,gamma,y = [],py = [],path = 'sp_input.json'):
    assert(len(x) == len(px))
    assert(len(x) == len(gamma))
    with open(path) as f:
        input_file = json.load(f,object_pairs_hook=OrderedDict)
    input_file['simulation']['dimension'] = 1
    input_file['beam']['x'] = list(x)
    input_file['beam']['px'] = list(px)
    input_file['beam']['gamma'] = list(gamma)
    if len(y) > 0:
        assert(len(y) == len(x))
        assert(len(py) == len(x))
        input_file['simulation']['dimension'] = 2
    input_file['beam']['y'] = list(y)
    input_file['beam']['py'] = list(py)
    with open(path,'w') as outfile:
        json.dump(input_file,outfile,indent=4)
    
def plot_init_phase_space(path = 'sp_input.json',s=1):
    with open(path) as f:
        input_file = json.load(f,object_pairs_hook=OrderedDict)
    x = input_file['beam']['x']
    px = input_file['beam']['px']
    plt.scatter(x,px,s=s)
    plt.xlabel('$x\;\;(c/\omega_{p0})$')
    plt.ylabel('$p_x$')
    plt.title('Initial phase space')
    plt.show()

### Computation

def Gaussian_focus_1D(x,A0,sig_i):
    if x == 0:
        return 0
    return A0 * sig_i ** 2 * (1 - np.exp(-x**2 / 2 / sig_i**2)) / x

def Gaussian_focus(x,y,A0,sig_i,direction):
    if x ** 2 + y ** 2 == 0:
        return 0
    if direction == 'x':
        return A0 * sig_i ** 2 * (1 - np.exp(-(x**2 + y**2) / 2 / sig_i**2)) / (x**2 + y**2 ) * x
    else:
        return A0 * sig_i ** 2 * (1 - np.exp(-(x**2 + y**2) / 2 / sig_i**2)) / (x**2 + y**2 ) * y
    
def ion_cylinder_focus_1D(x,A0,R):
    if abs(x) < R:
        return A0 * x / 2
    else:
        return A0 * R ** 2 / 2 / x
    
def ion_cylinder_focus(x,y,A0,R,direction):
    r2 = x ** 2 + y ** 2
    if r2 == 0:
        return 0
    if r2 <= R ** 2:
        if direction == 'x':
            return A0 * x / 2
        else:
            return A0 * y / 2
    else:
        if direction == 'x':
            return A0 * R ** 2 / 2 / r2 * x
        else:
            return A0 * R ** 2 / 2 / r2 * y

##### To be used by solve_ivp #####

def derivative(z,yvec,gamma_init,gamma_gain_model,A_model,sig_i_model,fs_model,ion_model = 'Gaussian'): 
    dimension = len(yvec) // 2
    if dimension == 1:
        x,px = yvec
    else:
        x,px,y,py = yvec
        
    local_density = float(fs_model(z))
    A = float(A_model(z))
    sig_i = float(sig_i_model(z))
    gamma_gain = float(gamma_gain_model(z))
    gamma = gamma_init + gamma_gain
    
    if ion_model == 'cylinder':
        if dimension == 1:
            dydz = [px/gamma,-x/2 * local_density - ion_cylinder_focus_1D(x,A,sig_i)]
        else:
            dydz = [px/gamma,-x/2 * local_density - ion_cylinder_focus(x,y,A,sig_i,'x'),\
                    py/gamma,-y/2 * local_density - ion_cylinder_focus(x,y,A,sig_i,'y')]
            
    elif ion_model == 'Gaussian':
        if dimension == 1:
            dydz = [px/gamma,-x/2 * local_density - Gaussian_focus_1D(x,A,sig_i)]
        else:
            dydz = [px/gamma,-x/2 * local_density - Gaussian_focus(x,y,A,sig_i,'x'),\
                    py/gamma,-y/2 * local_density - Gaussian_focus(x,y,A,sig_i,'y')]
    else:
        print('Wrong argument for ion_model!')
    return dydz

def derivative2(z,yvec,gamma_init,gamma_gain_model,focusing_field_model): 
    dimension = len(yvec) // 2
    if dimension == 1:
        x,px = yvec
        r = abs(x)
    else:
        x,px,y,py = yvec
        r = np.sqrt(x**2 + y**2)
        
    gamma_gain = float(gamma_gain_model(z))
    gamma = gamma_init + gamma_gain
    
    
    focusing_field_at_r = float(focusing_field_model(z,r))
    if dimension == 1:
        dydz = [px/gamma,-focusing_field_at_r]
    else:
        dydz = [px/gamma,-focusing_field_at_r / r * x,\
                py/gamma,-focusing_field_at_r / r * y]
    return dydz

def get_start_and_end_index(N_global,nprocs,rank):

    N_local,res = divmod(N_global,nprocs)
    if rank < res:
        N_local += 1
        start_idx = rank * N_local
        end_idx = start_idx + N_local
    else:
        start_idx = res * (N_local + 1) + (rank - res) * N_local
        end_idx = start_idx + N_local
    
    return start_idx,end_idx

### Visualization

def get_one_period(x,vx,z):
    start_idx = -1
    end_idx = -1
    assert(len(x) == len(vx))
    assert(len(x) == len(z))
    # Find start_idx, where the first time vx goes from >  0 to <= 0  :
    # vx[start_idx] <= 0 and vx[start_idx - 1] > 0
    
    for i in range(len(x)):
        if (i == 0 and vx[i] == 0) or (i > 0 and vx[i] <= 0 and vx[i-1] > 0):
            start_idx = i
            break
    # Find end_idx, where the first time vx goes from > 0 to <= 0 AFTER start_idx
    for i in range(start_idx+1,len(x)):
        if vx[i] <= 0 and vx[i-1] > 0:
            end_idx = i
            break
    if start_idx == -1 or end_idx == -1:
        print('Period not found!')
        print(start_idx)
        print(end_idx)
        return x,vx,z
    
    return x[start_idx:end_idx+1],vx[start_idx:end_idx+1],z[start_idx:end_idx+1]

# def get_time(x,vx):
#     assert(len(x) == len(vx))
#     dx = x[1:] - x[:-1]
#     vx_avg_in_dx = (vx[:-1] + vx[1:])/2 
#     dt = dx / vx_avg_in_dx
#     assert(all(dt >= 0))
#     return dt.sum()

def get_time(x,vx):
    assert(len(x) == len(vx))
    x = np.array(x)
    vx = np.array(vx)
    dx = np.abs(x[1:] - x[:-1])
    vx_avg = (np.abs(vx[1:]) + np.abs(vx[:-1]))/2
    assert all(vx_avg > 0)
    dt = dx/vx_avg
    T = dt.sum()
    if vx[-1] != 0:
        T = T + abs(x[-1] - x[0]) / abs(vx[-1])
    return T

def get_action(x,px):
    assert(len(x) == len(px))
    x = np.array(x)
    px = np.array(px)
    trapezoids_areas = (px[:-1] + px[1:]) * (x[1:] - x[:-1]) / 2
    # Only when px flip sign, the area of the trapezoid can be negative.
    # The calculation is still correct in this corner case.
    return trapezoids_areas.sum()

# Scatter plot for particles. Particles in different groups are in different colors.
# e.g. if len(x) = 10, indices_start = [0,3,6] (the first one has to be 0), colors = ['r','k','b'],
# then the particles with indices 0~2 will be red, 3~5 with be black, 6~last (9) will be in blue

def plot_beam_particles_diff_groups(x,px,indices_start,colors):
    assert(len(indices_start) == len(colors))
    nGroups = len(colors)
    indices_start.append(len(x))
    for i in range(nGroups):
        idx_start = indices_start[i]
        idx_start_next = indices_start[i+1]
        plt.scatter(x[idx_start : idx_start_next], x[idx_start : idx_start_next],color = colors[i])
    plt.xlabel('$x\;\;(c/\omega_{p0})$')
    plt.ylabel('$p_x$')
    plt.show()

def beam_raw_data_analysis(x,px):
    # The row indices are particle indices
    # The column indices are time steps
    # x.shape = px.shape = (# of particles, # of time steps)
    x_c = np.mean(x, axis = 0)
    x = x - x_c
    px_c = np.mean(px, axis = 0)
    px = px - px_c
    x2_avg = np.mean(x ** 2, axis = 0)
    px2_avg = np.mean(px ** 2, axis = 0)
    x_times_px_avg = np.mean(x * px, axis = 0)
    
    sigma_x = np.sqrt(x2_avg)
    sigma_px = np.sqrt(px2_avg)
    
    emittance_x = np.sqrt(x2_avg * px2_avg - x_times_px_avg ** 2)
    
    dic = {}
    dic['emittance'] = emittance_x
    dic['sigma'] = sigma_x
    dic['sigma_p'] = sigma_px
    dic['x_times_px_avg'] = x_times_px_avg
    dic['sigma_square'] = x2_avg
    dic['sigma_p_square'] = px2_avg
    return dic




def get_lineout_idx(Min,Max,lineout_pos,n_grid_points):
    if lineout_pos < Min or lineout_pos > Max:
        print('Lineout position is out of range:[',Min,',',Max,']!')
        return 0
    return int((n_grid_points-1) / (Max - Min) * (lineout_pos - Min) + 0.5)

def select_lineout_range(x,lineout,x_min,x_max):
    if x_min > x_max:
        print('Invalid lineout range!')
        return x,lineout

    x_min = max(x_min,x[0])
    x_max = min(x_max,x[-1])
    x_min_idx = get_lineout_idx(x[0],x[-1],x_min,len(x)) 
    x_max_idx = get_lineout_idx(x[0],x[-1],x_max,len(x))
    return x[x_min_idx:x_max_idx+1],lineout[x_min_idx:x_max_idx+1]

def get_lineout(filename,direction,lineout_pos,lineout_range = [-float('inf'),float('inf')],code = 'QPAD'):
    with h5py.File(filename, 'r') as h5file:
        dset_name = list(h5file.keys())[1] # dset_name = 'charge_slice_xz'
        data = np.array(h5file[dset_name])
        n_grid_points_xi, n_grid_points_x = data.shape
        x_range = np.array(h5file['AXIS']['AXIS1'])
        xi_range = np.array(h5file['AXIS']['AXIS2'])

    if direction == 'transverse':
        lineout_idx_xi = get_lineout_idx(xi_range[0],xi_range[1],lineout_pos,n_grid_points_xi)
        lineout = data[lineout_idx_xi,:]
        if code == 'QuickPIC':
            lineout = lineout[1:] # get rid of the empty data in the first column

        x = np.linspace(x_range[0],x_range[1],num = len(lineout))
        return select_lineout_range(x,lineout,lineout_range[0],lineout_range[1])
    elif direction == 'longitudinal':
        lineout_idx_x = get_lineout_idx(x_range[0],x_range[1],lineout_pos,n_grid_points_x)
        lineout = data[:,lineout_idx_x]

        xi = np.linspace(xi_range[0],xi_range[1],num = len(lineout))
        return select_lineout_range(xi,lineout,lineout_range[0],lineout_range[1])