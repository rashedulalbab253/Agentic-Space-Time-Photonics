import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def find_row_index(grid_z: np.ndarray, z_target: float) -> int:
    """Return index in grid_z closest to z_target (microns)."""
    return int(np.argmin(np.abs(grid_z - z_target)))


def central_lobe_mask(grid_x: np.ndarray, intensity_row: np.ndarray) -> np.ndarray:
    """Return boolean mask selecting samples belonging to the central lobe.

    The central lobe is defined here as the contiguous region around x=0 where the
    intensity is above half its maximum (FWHM region). Modify as needed.
    """
    half_max = intensity_row.max() * 0.5
    above = intensity_row >= half_max

    # Take the contiguous block that includes x~=0
    center_idx = int(np.argmin(np.abs(grid_x)))
    if not above[center_idx]:
        # If the absolute peak is not at x=0, shift to the global maximum
        center_idx = int(np.argmax(intensity_row))

    # Expand left
    left = center_idx
    while left > 0 and above[left - 1]:
        left -= 1

    # Expand right
    right = center_idx
    while right < len(grid_x) - 1 and above[right + 1]:
        right += 1

    mask = np.zeros_like(intensity_row, dtype=bool)
    mask[left : right + 1] = True
    return mask


def compute_ratio(intensity_row: np.ndarray, grid_x: np.ndarray) -> float:
    """Return central-lobe energy divided by total energy for one z slice."""
    total = np.trapz(intensity_row, grid_x)
    mask = central_lobe_mask(grid_x, intensity_row)
    central = np.trapz(intensity_row[mask], grid_x[mask])
    return central / total if total != 0 else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze far-field intensity slices.")
    parser.add_argument("--intensity", required=True, type=Path, help="Path to .npy array containing intensity |Hz|^2 (shape: [nz, nx])")
    parser.add_argument("--grid_z", required=True, type=Path, help="Path to .npy array of z coordinates (µm)")
    parser.add_argument("--grid_x", required=True, type=Path, help="Path to .npy array of x coordinates (µm)")
    parser.add_argument("--z", default=50.0, type=float, help="z position (µm) at which to evaluate (default: 50)")
    args = parser.parse_args()

    intensity = np.load(args.intensity)
    grid_z = np.load(args.grid_z)
    grid_x = np.load(args.grid_x)

    output_png = Path(args.intensity).with_suffix('.cross_section.png')

    row_idx = find_row_index(grid_z, args.z)
    intensity_row = intensity[row_idx]

    ratio = compute_ratio(intensity_row, grid_x)
    print(f"Energy ratio at z={grid_z[row_idx]:.2f} µm → {ratio:.4f}")

    # Visualisation: cross-section and central-lobe markers
    mask = central_lobe_mask(grid_x, intensity_row)
    left_edge = grid_x[mask][0]
    right_edge = grid_x[mask][-1]

    plt.figure(figsize=(8, 4))
    plt.plot(grid_x, intensity_row, label='Intensity profile')
    plt.axvline(left_edge, color='red', linestyle='--', label='Lobe edges')
    plt.axvline(right_edge, color='red', linestyle='--')
    plt.fill_between(grid_x, 0, intensity_row, where=mask, color='red', alpha=0.2)
    plt.title(f'Cross-section at z={grid_z[row_idx]:.2f} µm\nCentral-lobe energy ratio = {ratio:.3f}')
    plt.xlabel('x (µm)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_png, dpi=300)
    print(f"Plot saved to {output_png}")
    plt.show()


if __name__ == "__main__":
    main() 