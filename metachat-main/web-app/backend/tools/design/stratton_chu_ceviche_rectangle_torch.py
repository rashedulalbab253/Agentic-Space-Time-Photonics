import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np

def strattonChu2D(idx, dL, sx, sy, Nx, Ny, xc, yc, Rx, Ry, lambda_val, r_obs, Ex_OBJ, Ey_OBJ, Hz_OBJ, 
                  N_theta=361, debug=1, points_per_pixel=1, out_folder=".", device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert input fields to torch tensors on the given device
    # Ex_OBJ = torch.tensor(Ex_OBJ_np, dtype=torch.cfloat, device=device, requires_grad=True)
    # Ey_OBJ = torch.tensor(Ey_OBJ_np, dtype=torch.cfloat, device=device, requires_grad=True)
    # Hz_OBJ = torch.tensor(Hz_OBJ_np, dtype=torch.cfloat, device=device, requires_grad=True)

    # Split fields into real and imaginary parts
    Ex_real, Ex_imag = Ex_OBJ.real, Ex_OBJ.imag
    Ey_real, Ey_imag = Ey_OBJ.real, Ey_OBJ.imag
    Hz_real, Hz_imag = Hz_OBJ.real, Hz_OBJ.imag

    # Coordinates of the rectangle's corners
    x_left = xc - Rx
    x_right = xc + Rx
    y_bottom = yc - Ry
    y_top = yc + Ry

    vertical_dist = (y_top - y_bottom)
    horizontal_dist = (x_right - x_left)

    npx_per_vertical_edge = round(points_per_pixel * vertical_dist / dL)
    npx_per_horizontal_edge = round(points_per_pixel * horizontal_dist / dL)

    # Generate points for each edge
    X_bottom = torch.linspace(x_left, x_right, npx_per_horizontal_edge, device=device)
    Y_bottom = torch.full((npx_per_horizontal_edge,), y_bottom, device=device)
    Y_right = torch.linspace(y_bottom, y_top, npx_per_vertical_edge, device=device)
    X_right = torch.full((npx_per_vertical_edge,), x_right, device=device)
    X_top = torch.linspace(x_right, x_left, npx_per_horizontal_edge, device=device)
    Y_top = torch.full((npx_per_horizontal_edge,), y_top, device=device)
    Y_left = torch.linspace(y_top, y_bottom, npx_per_vertical_edge, device=device)
    X_left = torch.full((npx_per_vertical_edge,), x_left, device=device)

    # Combine the points
    X_surf = torch.cat([X_bottom, X_right, X_top, X_left], dim=0)
    Y_surf = torch.cat([Y_bottom, Y_right, Y_top, Y_left], dim=0)

    # Global grid
    y_linspace = torch.linspace(-sy / 2, sy / 2, Ny, device=device)
    x_linspace = torch.linspace(-sx / 2, sx / 2, Nx, device=device)
    X_glob, Y_glob = torch.meshgrid(x_linspace, y_linspace, indexing='xy')
    X_glob = X_glob.to(device)
    Y_glob = Y_glob.to(device)

    # Prepare fields for grid_sample
    Ex_real_in = Ex_real.unsqueeze(0).unsqueeze(0)  # [1, 1, Ny, Nx]
    Ex_imag_in = Ex_imag.unsqueeze(0).unsqueeze(0)
    Ey_real_in = Ey_real.unsqueeze(0).unsqueeze(0)
    Ey_imag_in = Ey_imag.unsqueeze(0).unsqueeze(0)
    Hz_real_in = Hz_real.unsqueeze(0).unsqueeze(0)
    Hz_imag_in = Hz_imag.unsqueeze(0).unsqueeze(0)

    # Normalize coordinates for grid_sample
    X_min, X_max = -sx / 2, sx / 2
    Y_min, Y_max = -sy / 2, sy / 2
    grid_x = 2 * (X_surf - X_min) / (X_max - X_min) - 1
    grid_y = 2 * (Y_surf - Y_min) / (Y_max - Y_min) - 1

    # Create a grid for interpolation
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(2).to(torch.float32)  # [1, N_points, 1, 2]

    # Interpolate real and imaginary parts separately
    Ex_surf_real = F.grid_sample(Ex_real_in, grid, align_corners=True).squeeze(0).squeeze(-1).squeeze(0)
    Ex_surf_imag = F.grid_sample(Ex_imag_in, grid, align_corners=True).squeeze(0).squeeze(-1).squeeze(0)
    Ey_surf_real = F.grid_sample(Ey_real_in, grid, align_corners=True).squeeze(0).squeeze(-1).squeeze(0)
    Ey_surf_imag = F.grid_sample(Ey_imag_in, grid, align_corners=True).squeeze(0).squeeze(-1).squeeze(0)
    Hz_surf_real = F.grid_sample(Hz_real_in, grid, align_corners=True).squeeze(0).squeeze(-1).squeeze(0)
    Hz_surf_imag = F.grid_sample(Hz_imag_in, grid, align_corners=True).squeeze(0).squeeze(-1).squeeze(0)

    # Combine real and imaginary parts
    Ex_surf = Ex_surf_real + 1j * Ex_surf_imag
    Ey_surf = Ey_surf_real + 1j * Ey_surf_imag
    Hz_surf = Hz_surf_real + 1j * Hz_surf_imag

    if debug:
        print("Interpolation complete")
        plotNearField(idx, out_folder, sx, sy, X_surf.detach(), Y_surf.detach(), 
                      X_glob.detach().T, Y_glob.detach().T, Ex_OBJ, Ex_surf, 'Ex')
        plotNearField(idx, out_folder, sx, sy, X_surf.detach(), Y_surf.detach(), 
                      X_glob.detach().T, Y_glob.detach().T, Ey_OBJ, Ey_surf, 'Ey')
        plotNearField(idx, out_folder, sx, sy, X_surf.detach(), Y_surf.detach(), 
                      X_glob.detach().T, Y_glob.detach().T, Hz_OBJ, Hz_surf, 'Hz')

    # Define surface normals
    nx_bottom = torch.zeros(npx_per_horizontal_edge, dtype=torch.cfloat, device=device)
    ny_bottom = -torch.ones(npx_per_horizontal_edge, dtype=torch.cfloat, device=device)
    nx_right = torch.ones(npx_per_vertical_edge, dtype=torch.cfloat, device=device)
    ny_right = torch.zeros(npx_per_vertical_edge, dtype=torch.cfloat, device=device)
    nx_top = torch.zeros(npx_per_horizontal_edge, dtype=torch.cfloat, device=device)
    ny_top = torch.ones(npx_per_horizontal_edge, dtype=torch.cfloat, device=device)
    nx_left = -torch.ones(npx_per_vertical_edge, dtype=torch.cfloat, device=device)
    ny_left = torch.zeros(npx_per_vertical_edge, dtype=torch.cfloat, device=device)

    nx = torch.cat([nx_bottom, nx_right, nx_top, nx_left])
    ny = torch.cat([ny_bottom, ny_right, ny_top, ny_left])

    # Calculate surface element size
    dx = (X_glob[0, 1] - X_glob[0, 0])
    dy = (Y_glob[1, 0] - Y_glob[0, 0])
    if not torch.isclose(dx, dy, atol=1e-5):
        raise ValueError("dx and dy must be equal")
    
    dl_surf = dx

    # Constants
    Z0 = 120 * torch.pi
    c = 299792458
    eps0 = 8.854187817e-12
    k0 = 2 * torch.pi / lambda_val

    theta_obs = torch.linspace(0, 2 * torch.pi, N_theta, device=device)

    Far_Ex = torch.zeros(N_theta, dtype=torch.cfloat, device=device)
    Far_Ey = torch.zeros(N_theta, dtype=torch.cfloat, device=device)
    Far_Hz = torch.zeros(N_theta, dtype=torch.cfloat, device=device)

    ux = torch.cos(theta_obs).unsqueeze(1)
    uy = torch.sin(theta_obs).unsqueeze(1)

    r_rs = torch.sqrt((r_obs * ux - X_surf)**2 + (r_obs * uy - Y_surf)**2)
    phas_kr = torch.exp(-1j * k0 * r_rs) / (4 * torch.pi * r_rs)

    Far_Ex = torch.sum(1j * k0 * (Z0 * ny * Hz_surf - (nx * Ey_surf - ny * Ex_surf) * uy + 
                    (nx * Ex_surf + ny * Ey_surf) * ux) * phas_kr * dl_surf, dim=1)
    Far_Ey = torch.sum(1j * k0 * (-Z0 * nx * Hz_surf + (nx * Ey_surf - ny * Ex_surf) * ux + 
                    (nx * Ex_surf + ny * Ey_surf) * uy) * phas_kr * dl_surf, dim=1)
    Far_Hz = -torch.sum(1j * k0 * (c * eps0 * (nx * Ey_surf - ny * Ex_surf) - 
                    ((ny * Hz_surf) * uy + (nx * Hz_surf) * ux)) * phas_kr * dl_surf, dim=1)

    # fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 12))
    
    # # Ex subplots
    # ax1.plot(theta_obs.detach().cpu().numpy(), np.real(Far_Ex.detach().cpu().numpy()))
    # ax1.grid(True)
    # ax1.set_ylabel('Ex Amplitude')
    # ax1.set_title('Real(Ex)')
    
    # ax2.plot(theta_obs.detach().cpu().numpy(), np.imag(Far_Ex.detach().cpu().numpy()))
    # ax2.grid(True)
    # ax2.set_title('Imaginary(Ex)')
    
    # # Ey subplots
    # ax3.plot(theta_obs.detach().cpu().numpy(), np.real(Far_Ey.detach().cpu().numpy()))
    # ax3.grid(True)
    # ax3.set_ylabel('Ey Amplitude')
    # ax3.set_title('Real(Ey)')
    
    # ax4.plot(theta_obs.detach().cpu().numpy(), np.imag(Far_Ey.detach().cpu().numpy()))
    # ax4.grid(True)
    # ax4.set_title('Imaginary(Ey)')
    
    # # Hz subplots
    # ax5.plot(theta_obs.detach().cpu().numpy(), np.real(Far_Hz.detach().cpu().numpy()))
    # ax5.grid(True)
    # ax5.set_xlabel('Theta')
    # ax5.set_ylabel('Hz Amplitude')
    # ax5.set_title('Real(Hz)')
    
    # ax6.plot(theta_obs.detach().cpu().numpy(), np.imag(Far_Hz.detach().cpu().numpy()))
    # ax6.grid(True)
    # ax6.set_xlabel('Theta')
    # ax6.set_title('Imaginary(Hz)')
    
    # plt.tight_layout()
    # plt.savefig('new_fars.png')
    # plt.close()

    # import pdb; pdb.set_trace()

    if debug:
        print("Far field calculation complete (Stratton-Chu)")

    return theta_obs, Far_Ex, Far_Ey, Far_Hz


def strattonChu2D_batched(dL, sx, sy, Nx, Ny, xc, yc, Rx, Ry, lambda_val, r_obs,
                          Ex_OBJ, Ey_OBJ, Hz_OBJ, N_theta=361, debug=1, points_per_pixel=1, device=None):

    if device is None:
        device = Ex_OBJ.device

    B, Ny_field, Nx_field = Ex_OBJ.shape

    # Split fields into real and imaginary parts
    Ex_real, Ex_imag = Ex_OBJ.real, Ex_OBJ.imag
    Ey_real, Ey_imag = Ey_OBJ.real, Ey_OBJ.imag
    Hz_real, Hz_imag = Hz_OBJ.real, Hz_OBJ.imag

    x_left = xc - Rx
    x_right = xc + Rx
    y_bottom = yc - Ry
    y_top = yc + Ry

    vertical_dist = (y_top - y_bottom)
    horizontal_dist = (x_right - x_left)

    npx_per_vertical_edge = round(points_per_pixel * vertical_dist / dL)
    npx_per_horizontal_edge = round(points_per_pixel * horizontal_dist / dL)

    # Generate points for each edge (no batch dimension needed)
    X_bottom = torch.linspace(x_left, x_right, npx_per_horizontal_edge, device=device)
    Y_bottom = torch.full((npx_per_horizontal_edge,), y_bottom, device=device)
    Y_right = torch.linspace(y_bottom, y_top, npx_per_vertical_edge, device=device)
    X_right = torch.full((npx_per_vertical_edge,), x_right, device=device)
    X_top = torch.linspace(x_right, x_left, npx_per_horizontal_edge, device=device)
    Y_top = torch.full((npx_per_horizontal_edge,), y_top, device=device)
    Y_left = torch.linspace(y_top, y_bottom, npx_per_vertical_edge, device=device)
    X_left = torch.full((npx_per_vertical_edge,), x_left, device=device)

    X_surf = torch.cat([X_bottom, X_right, X_top, X_left], dim=0)  # [N_points]
    Y_surf = torch.cat([Y_bottom, Y_right, Y_top, Y_left], dim=0)  # [N_points]
    N_points = X_surf.shape[0]

    y_linspace = torch.linspace(-sy / 2, sy / 2, Ny, device=device)
    x_linspace = torch.linspace(-sx / 2, sx / 2, Nx, device=device)
    X_glob, Y_glob = torch.meshgrid(x_linspace, y_linspace, indexing='xy')
    # X_glob, Y_glob: [Nx, Ny]

    # Prepare fields for grid_sample: we need [B,1,Ny_field,Nx_field]
    Ex_real_in = Ex_real.unsqueeze(1)  # [B,1,Ny_field,Nx_field]
    Ex_imag_in = Ex_imag.unsqueeze(1)
    Ey_real_in = Ey_real.unsqueeze(1)
    Ey_imag_in = Ey_imag.unsqueeze(1)
    Hz_real_in = Hz_real.unsqueeze(1)
    Hz_imag_in = Hz_imag.unsqueeze(1)

    X_min, X_max = -sx / 2, sx / 2
    Y_min, Y_max = -sy / 2, sy / 2
    grid_x = 2 * (X_surf - X_min) / (X_max - X_min) - 1  # [N_points]
    grid_y = 2 * (Y_surf - Y_min) / (Y_max - Y_min) - 1  # [N_points]

    # Create grid: [B, N_points, 1, 2]
    # Repeat grid for each sample in batch
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)  # [N_points,1,2]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1).to(torch.float32)  # [B,N_points,1,2]

    # Interpolate
    def interpolate_field(real_in, imag_in):
        real_out = F.grid_sample(real_in, grid, align_corners=True)  # [B,1,N_points,1]
        imag_out = F.grid_sample(imag_in, grid, align_corners=True)
        real_out = real_out.squeeze(-1).squeeze(1)  # [B,N_points]
        imag_out = imag_out.squeeze(-1).squeeze(1)
        return real_out + 1j * imag_out

    Ex_surf = interpolate_field(Ex_real_in, Ex_imag_in)  # [B,N_points]
    Ey_surf = interpolate_field(Ey_real_in, Ey_imag_in)  # [B,N_points]
    Hz_surf = interpolate_field(Hz_real_in, Hz_imag_in)  # [B,N_points]

    if debug:
        print("Interpolation complete (batched).")

    # Normals
    nx_bottom = torch.zeros(npx_per_horizontal_edge, dtype=torch.cfloat, device=device)
    ny_bottom = -torch.ones(npx_per_horizontal_edge, dtype=torch.cfloat, device=device)
    nx_right = torch.ones(npx_per_vertical_edge, dtype=torch.cfloat, device=device)
    ny_right = torch.zeros(npx_per_vertical_edge, dtype=torch.cfloat, device=device)
    nx_top = torch.zeros(npx_per_horizontal_edge, dtype=torch.cfloat, device=device)
    ny_top = torch.ones(npx_per_horizontal_edge, dtype=torch.cfloat, device=device)
    nx_left = -torch.ones(npx_per_vertical_edge, dtype=torch.cfloat, device=device)
    ny_left = torch.zeros(npx_per_vertical_edge, dtype=torch.cfloat, device=device)

    nx = torch.cat([nx_bottom, nx_right, nx_top, nx_left])  # [N_points]
    ny = torch.cat([ny_bottom, ny_right, ny_top, ny_left])  # [N_points]

    dx = (X_glob[0, 1] - X_glob[0, 0])
    dy = (Y_glob[1, 0] - Y_glob[0, 0])
    if not torch.isclose(dx, dy, atol=1e-5):
        raise ValueError("dx and dy must be equal")
    dl_surf = dx

    Z0 = 120 * torch.pi
    c = 299792458
    eps0 = 8.854187817e-12
    k0 = 2 * torch.pi / lambda_val

    theta_obs = torch.linspace(0, 2 * torch.pi, N_theta, device=device)  # [N_theta]

    # Shape management:
    # Ex_surf, Ey_surf, Hz_surf: [B,N_points]
    # nx, ny: [N_points]
    # ux, uy: [N_theta]
    # We want broadcasting: [B,N_theta,N_points]
    ux = torch.cos(theta_obs).unsqueeze(0).unsqueeze(-1)  # [1,N_theta,1]
    uy = torch.sin(theta_obs).unsqueeze(0).unsqueeze(-1)  # [1,N_theta,1]

    # Expand to [B,N_theta,N_points]
    ux = ux.expand(B, -1, N_points)
    uy = uy.expand(B, -1, N_points)

    X_surf_b = X_surf.unsqueeze(0).expand(B, -1)  # [B,N_points]
    Y_surf_b = Y_surf.unsqueeze(0).expand(B, -1)  # [B,N_points]

    r_rs = torch.sqrt((r_obs * ux - X_surf_b.unsqueeze(1))**2 + (r_obs * uy - Y_surf_b.unsqueeze(1))**2)
    # r_rs: [B,N_theta,N_points]

    phas_kr = torch.exp(-1j * k0 * r_rs) / (4 * torch.pi * r_rs)

    # Expand nx, ny for batch and theta:
    nx_b = nx.unsqueeze(0).expand(B, -1)  # [B,N_points]
    ny_b = ny.unsqueeze(0).expand(B, -1)  # [B,N_points]

    # Expand to [B,1,N_points] for broadcast with [B,N_theta,N_points]
    nx_b = nx_b.unsqueeze(1)
    ny_b = ny_b.unsqueeze(1)

    Ex_surf = Ex_surf.unsqueeze(1)  # [B,1,N_points]
    Ey_surf = Ey_surf.unsqueeze(1)
    Hz_surf = Hz_surf.unsqueeze(1)

    # Equations as in code, now with batch dimension:
    Far_Ex = torch.sum(1j * k0 * (Z0 * ny_b * Hz_surf - (nx_b * Ey_surf - ny_b * Ex_surf) * uy +
                    (nx_b * Ex_surf + ny_b * Ey_surf) * ux) * phas_kr * dl_surf, dim=2)
    Far_Ey = torch.sum(1j * k0 * (-Z0 * nx_b * Hz_surf + (nx_b * Ey_surf - ny_b * Ex_surf) * ux +
                    (nx_b * Ex_surf + ny_b * Ey_surf) * uy) * phas_kr * dl_surf, dim=2)
    Far_Hz = torch.sum(-1j * k0 * (c * eps0 * (nx_b * Ey_surf - ny_b * Ex_surf) -
                    ((ny_b * Hz_surf) * uy + (nx_b * Hz_surf) * ux)) * phas_kr * dl_surf, dim=2)

    # Far_Ex, Far_Ey, Far_Hz: [B,N_theta]

    if debug:
        print("Far field calculation complete (Stratton-Chu, batched)")

    return theta_obs, Far_Ex, Far_Ey, Far_Hz