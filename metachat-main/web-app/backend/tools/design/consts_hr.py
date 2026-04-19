import numpy as np

## Optimization stuff
opt_iters = 50

## Simulation setup stuff

eps_mat = 5.76
n_sub = 1.44

thickness = 50

N_bars = 10

Nx = 500
Ny = 350
w_min = 3
pml_x = 60
pml_y = 60

image_sizex = 180 # 256 # 96
image_sizey = 55 # 32

eps_0 = 8.85418782e-12
mu_0 = 1.25663706e-6
C_0 = 1 / np.sqrt(eps_0 * mu_0)  # speed of light in vacuum

dL = 10
wl = 5.5e-07

device_length = max(image_sizex, image_sizey)
# spacing = int((Nx-2*pml_x-device_length)/2)-15
spacing=85
adj_src_loc_y = pml_y+spacing+image_sizey
# These are the bounds of the structure (i.e., the dielectric region that needs to be optimized)
x_start, x_end = int(Nx/2-image_sizex/2), int(Nx/2-image_sizex/2)+image_sizex
y_start, y_end = pml_y+spacing, pml_y+spacing+image_sizey

adj_src_loc_y = pml_y+spacing+image_sizey

## Near to far field stuff
N_theta = 360*10+1

nn_x_pix = 256
nn_y_pix = 128

nn_padding = 3