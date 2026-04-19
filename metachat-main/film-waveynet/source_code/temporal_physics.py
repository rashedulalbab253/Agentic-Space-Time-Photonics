"""
Physics-informed temporal loss terms for FiLM WaveY-Net.

Implements two key constraints for temporally-modulated metasurfaces:
1. Continuity of D (= eps * E) and B (= mu * H) at temporal switching boundaries (Eq. 3)
2. Frozen eigenmode condition during inductive state — spatial profile is preserved,
   only amplitude may change.

These are added as soft penalty terms during training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import consts


def continuity_loss_DB(fields_before, fields_after, eps_before, eps_after,
                       mu_0=consts.mu_0):
    """
    Enforce continuity of D and B fields at temporal switching boundaries.

    At a temporal boundary (instantaneous material switch), Maxwell's equations
    require that the displacement field D = eps * E and the magnetic flux
    density B = mu * H are continuous across the boundary.

    Parameters
    ----------
    fields_before : torch.Tensor
        Predicted fields at time just before switching, shape (B, 2, H, W).
        Channel 0 = Hz_real, Channel 1 = Hz_imag.
    fields_after : torch.Tensor
        Predicted fields at time just after switching, shape (B, 2, H, W).
    eps_before : torch.Tensor
        Permittivity distribution before switching, shape (B, 1, H, W).
    eps_after : torch.Tensor
        Permittivity distribution after switching, shape (B, 1, H, W).
    mu_0 : float
        Vacuum permeability (constant for non-magnetic media).

    Returns
    -------
    loss : torch.Tensor
        Scalar loss penalizing discontinuities in D and B.
    """
    # --- B-field continuity: B = mu_0 * H ---
    # For non-magnetic media mu doesn't change, so H must be continuous.
    # Hz_before == Hz_after  =>  B continuity
    B_loss = F.mse_loss(
        mu_0 * fields_before,
        mu_0 * fields_after
    )

    # --- D-field continuity: D = eps * E ---
    # We work with Hz fields. From Hz we can infer the constraint on the
    # electric fields E. Since E ~ (1/eps) * curl(H), continuity of D = eps*E
    # means curl(H) must be continuous. We enforce this via finite differences
    # on the Hz field weighted by the permittivity.
    #
    # D_x = eps * E_x  and  E_x ~ (dHz/dy) / (j*omega*eps)
    # => D_x ~ dHz/dy / (j*omega)  (eps cancels)
    # So D_x continuity reduces to continuity of dHz/dy, which is spatial
    # derivative of the H field — must be continuous across temporal boundary.
    #
    # We approximate this by penalizing the difference in spatial gradients
    # of Hz across the temporal boundary.

    # Spatial gradients along y (dim=-2) and x (dim=-1)
    dHz_dy_before = fields_before[:, :, 1:, :] - fields_before[:, :, :-1, :]
    dHz_dy_after = fields_after[:, :, 1:, :] - fields_after[:, :, :-1, :]
    dHz_dx_before = fields_before[:, :, :, 1:] - fields_before[:, :, :, :-1]
    dHz_dx_after = fields_after[:, :, :, 1:] - fields_after[:, :, :, :-1]

    D_loss = (
        F.mse_loss(dHz_dy_before, dHz_dy_after) +
        F.mse_loss(dHz_dx_before, dHz_dx_after)
    )

    return B_loss + D_loss


def frozen_eigenmode_loss(fields_t1, fields_t2, is_inductive_mask):
    """
    Enforce frozen eigenmode condition during the inductive state.

    During the inductive state, the eigenmode spatial profile should remain
    unchanged — only the overall amplitude may vary. This is enforced by
    normalizing the fields and penalizing changes in the normalized profile.

    Parameters
    ----------
    fields_t1 : torch.Tensor
        Fields at time t1, shape (B, 2, H, W).
    fields_t2 : torch.Tensor
        Fields at time t2, shape (B, 2, H, W).
    is_inductive_mask : torch.Tensor
        Boolean mask of shape (B,), True for samples in inductive state.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss penalizing eigenmode profile changes during inductive state.
        Returns 0 if no samples are in inductive state.
    """
    if not is_inductive_mask.any():
        return torch.tensor(0.0, device=fields_t1.device)

    # Select inductive samples
    f1 = fields_t1[is_inductive_mask]  # (N_ind, 2, H, W)
    f2 = fields_t2[is_inductive_mask]

    # Compute L2 norm per sample for normalization
    norm1 = torch.norm(f1.reshape(f1.shape[0], -1), dim=1, keepdim=True)
    norm2 = torch.norm(f2.reshape(f2.shape[0], -1), dim=1, keepdim=True)

    # Avoid division by zero
    norm1 = torch.clamp(norm1, min=1e-10)
    norm2 = torch.clamp(norm2, min=1e-10)

    # Normalize: preserve spatial profile, remove amplitude
    f1_normalized = f1.reshape(f1.shape[0], -1) / norm1
    f2_normalized = f2.reshape(f2.shape[0], -1) / norm2

    # Penalize differences in normalized spatial profiles
    loss = F.mse_loss(f1_normalized, f2_normalized)

    return loss


def temporal_physics_loss(model, sample_batched, pattern, args,
                          field_scaling_factor, src_data_scaling_factor,
                          dt=0.05):
    """
    Compute combined temporal physics-informed loss.

    Evaluates the model at the current time t and at t + dt, then computes:
      1. D/B continuity loss at the switching boundary
      2. Frozen eigenmode loss for inductive-state samples

    Parameters
    ----------
    model : nn.Module
        The FiLM WaveY-Net model.
    sample_batched : dict
        Current training batch with keys: structure, field, src, wavelength_normalized,
        angle_normalized, time_state, time_state_normalized.
    pattern : torch.Tensor
        Permittivity pattern (with substrate row prepended), shape (B, 1, H+1, W).
    args : Namespace
        Training configuration with lambda_continuity, lambda_frozen_mode.
    field_scaling_factor : float
        Scaling factor for field normalization.
    src_data_scaling_factor : float
        Scaling factor for source data normalization.
    dt : float
        Temporal step (in normalized time units) for evaluating continuity.

    Returns
    -------
    total_loss : torch.Tensor
        Weighted sum of continuity and frozen-eigenmode losses.
    loss_dict : dict
        Individual loss components for logging.
    """
    device = next(model.parameters()).device

    x_batch = sample_batched['structure'].to(device)
    src_batch = sample_batched['src'].to(device)
    wavelengths_normalized = sample_batched['wavelength_normalized'].to(device)
    angles_normalized = sample_batched['angle_normalized'].to(device)
    time_state_normalized = sample_batched['time_state_normalized'].to(device)
    time_state = sample_batched['time_state'].to(device)

    x = x_batch[:, :, 1:-1, 1:-1]
    src_input = src_batch[:, :, 1:-1, 1:-1] / src_data_scaling_factor
    model_input = torch.cat((x, src_input), dim=1)

    # --- Evaluate model at current time t ---
    fields_t = model(model_input, wavelengths_normalized, angles_normalized, time_state_normalized)

    # --- Evaluate model at t + dt (clamped to [0, 1]) ---
    time_state_shifted = torch.clamp(time_state_normalized + dt, 0.0, 1.0)
    fields_t_dt = model(model_input, wavelengths_normalized, angles_normalized, time_state_shifted)

    # --- Construct permittivity before/after ---
    # The permittivity change at the switching boundary is modeled as a function
    # of the time state. For simplicity, we use the structure pattern itself,
    # scaling by 1.0 at t and a modulated value at t+dt.
    eps_before = pattern[:, :, 1:, :]  # trim substrate row
    # After switching, permittivity is modulated — slight change
    eps_after = eps_before  # In non-magnetic media, same structure

    # --- Continuity loss ---
    lambda_continuity = getattr(args, 'lambda_continuity', 0.1)
    L_continuity = continuity_loss_DB(fields_t, fields_t_dt, eps_before, eps_after)

    # --- Frozen eigenmode loss ---
    # Determine which samples are in the inductive state.
    # Convention: time_state < 0.5 => capacitive, >= 0.5 => inductive
    # (This threshold should be adapted to the specific switching protocol.)
    is_inductive = time_state_normalized >= 0.5
    lambda_frozen = getattr(args, 'lambda_frozen_mode', 0.05)
    L_frozen = frozen_eigenmode_loss(fields_t, fields_t_dt, is_inductive)

    total_loss = lambda_continuity * L_continuity + lambda_frozen * L_frozen

    loss_dict = {
        'continuity_loss': L_continuity.item(),
        'frozen_eigenmode_loss': L_frozen.item(),
        'temporal_total_loss': total_loss.item()
    }

    return total_loss, loss_dict
