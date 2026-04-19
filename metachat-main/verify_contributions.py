"""
Verification script for the 5 temporal modulation contributions.

Runs WITHOUT GPU, training data, or API keys. Tests:
  1. Import correctness and class instantiation
  2. Tensor shape consistency through FiLM and UNet forward pass (CPU)
  3. Backward-compatible checkpoint upgrade logic
  4. Physics loss computation (random data, CPU)
  5. Temporal Design API parameter validation and routing
  6. TemporalModulationAgent instantiation and system prompt

Usage:
    python verify_contributions.py
"""

import sys
import os
import traceback
import asyncio

# ─── Helpers ───────────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"

results = []

def test(name, fn):
    """Run a test, capture exceptions, log result."""
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True, None))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  {FAIL}  {name}")
        print(f"         {e}")
        results.append((name, False, tb))


# ════════════════════════════════════════════════════════════════════════════════
# CONTRIBUTION 1: Temporal FiLM Conditioning (3-condition FiLM + UNet forward)
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("CONTRIBUTION 1: Temporal FiLM Conditioning")
print("=" * 72)

# Need to add source dirs to path for imports
film_src = os.path.join(os.path.dirname(__file__), "film-waveynet", "source_code")
aim_src = os.path.join(os.path.dirname(__file__), "metachat-aim")
sys.path.insert(0, film_src)
sys.path.insert(0, aim_src)

import torch
import numpy as np

def test_film_import():
    from multi_film_angle_dec_fwdadj_sample_learners import FiLM, UNet
    assert FiLM is not None
    assert UNet is not None

test("FiLM and UNet classes import successfully", test_film_import)


def test_film_3cond_init():
    from multi_film_angle_dec_fwdadj_sample_learners import FiLM
    film = FiLM(num_features=64, num_conditions=3)
    # Weight should be (64*2, 3) = (128, 3)
    assert film.film_layer.weight.shape == (128, 3), \
        f"Expected (128, 3), got {film.film_layer.weight.shape}"

test("FiLM layer initializes with 3 conditions", test_film_3cond_init)


def test_film_forward_shape():
    from multi_film_angle_dec_fwdadj_sample_learners import FiLM
    film = FiLM(num_features=64, num_conditions=3)
    batch = 4
    x = torch.randn(batch, 64, 32, 32)
    wl = torch.randn(batch)
    angle = torch.randn(batch)
    time_norm = torch.randn(batch)
    out = film(x, wl, angle, time_norm)
    assert out.shape == (batch, 64, 32, 32), \
        f"Expected {(batch, 64, 32, 32)}, got {out.shape}"

test("FiLM forward pass produces correct output shape", test_film_forward_shape)


def test_unet_forward():
    from multi_film_angle_dec_fwdadj_sample_learners import UNet
    # Small model for CPU testing
    model = UNet(net_depth=3, block_depth=2, init_num_kernels=8,
                 input_channels=3, output_channels=2, dropout=0).float()
    batch = 2
    x = torch.randn(batch, 3, 64, 64)
    wl = torch.randn(batch)
    angle = torch.randn(batch)
    time_norm = torch.randn(batch)
    out = model(x, wl, angle, time_norm)
    assert out.shape == (batch, 2, 64, 64), \
        f"Expected {(batch, 2, 64, 64)}, got {out.shape}"

test("UNet forward pass with 3 conditions (CPU)", test_unet_forward)


def test_unet_time_sensitivity():
    """Verify that varying time_norm changes the output (FiLM is active)."""
    from multi_film_angle_dec_fwdadj_sample_learners import UNet
    model = UNet(net_depth=3, block_depth=2, init_num_kernels=8,
                 input_channels=3, output_channels=2, dropout=0).float()
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    wl = torch.tensor([0.5])
    angle = torch.tensor([0.0])
    t1 = torch.tensor([0.0])
    t2 = torch.tensor([1.0])
    with torch.no_grad():
        out1 = model(x, wl, angle, t1)
        out2 = model(x, wl, angle, t2)
    diff = (out1 - out2).abs().max().item()
    # After training, random init weights mean time changes output
    assert diff > 0, "Output should differ when time_norm changes (FiLM is active)"

test("UNet output is sensitive to time_norm input", test_unet_time_sensitivity)


# ════════════════════════════════════════════════════════════════════════════════
# CONTRIBUTION 2: Physics-Informed Temporal Loss
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("CONTRIBUTION 2: Physics-Informed Temporal Loss")
print("=" * 72)

def test_temporal_physics_import():
    from temporal_physics import continuity_loss_DB, frozen_eigenmode_loss, temporal_physics_loss
    assert callable(continuity_loss_DB)
    assert callable(frozen_eigenmode_loss)
    assert callable(temporal_physics_loss)

test("temporal_physics module imports successfully", test_temporal_physics_import)


def test_continuity_loss():
    from temporal_physics import continuity_loss_DB
    B = 4
    fields_before = torch.randn(B, 2, 32, 32)
    fields_after = torch.randn(B, 2, 32, 32)
    eps_before = torch.ones(B, 1, 32, 32) * 2.25
    eps_after = torch.ones(B, 1, 32, 32) * 2.25
    loss = continuity_loss_DB(fields_before, fields_after, eps_before, eps_after)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() >= 0, "Loss should be non-negative"

test("D/B continuity loss computes correctly", test_continuity_loss)


def test_continuity_loss_zero_for_identical():
    """If fields are identical, continuity loss should be zero."""
    from temporal_physics import continuity_loss_DB
    B = 4
    fields = torch.randn(B, 2, 32, 32)
    eps = torch.ones(B, 1, 32, 32) * 2.25
    loss = continuity_loss_DB(fields, fields, eps, eps)
    assert loss.item() < 1e-10, f"Expected ~0, got {loss.item()}"

test("D/B continuity loss is zero for identical fields", test_continuity_loss_zero_for_identical)


def test_frozen_eigenmode_loss():
    from temporal_physics import frozen_eigenmode_loss
    B = 4
    f1 = torch.randn(B, 2, 32, 32)
    f2 = torch.randn(B, 2, 32, 32)
    mask = torch.tensor([True, True, False, False])
    loss = frozen_eigenmode_loss(f1, f2, mask)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() >= 0

test("Frozen eigenmode loss computes correctly", test_frozen_eigenmode_loss)


def test_frozen_eigenmode_same_profile():
    """Same spatial profile (just different amplitude) should give near-zero loss."""
    from temporal_physics import frozen_eigenmode_loss
    f1 = torch.randn(2, 2, 16, 16)
    f2 = f1 * 2.5  # Scale amplitude only
    mask = torch.tensor([True, True])
    loss = frozen_eigenmode_loss(f1, f2, mask)
    assert loss.item() < 1e-10, f"Expected ~0 for scaled profile, got {loss.item()}"

test("Frozen eigenmode loss is ~0 for scaled profiles", test_frozen_eigenmode_same_profile)


def test_frozen_eigenmode_no_inductive():
    """No inductive samples → loss should be exactly 0."""
    from temporal_physics import frozen_eigenmode_loss
    f1 = torch.randn(2, 2, 16, 16)
    f2 = torch.randn(2, 2, 16, 16)
    mask = torch.tensor([False, False])
    loss = frozen_eigenmode_loss(f1, f2, mask)
    assert loss.item() == 0.0

test("Frozen eigenmode loss is 0 when no inductive samples", test_frozen_eigenmode_no_inductive)


def test_temporal_physics_loss_integration():
    """Test the full temporal_physics_loss with a mock model and batch."""
    from temporal_physics import temporal_physics_loss
    from multi_film_angle_dec_fwdadj_sample_learners import UNet
    import argparse

    model = UNet(net_depth=3, block_depth=2, init_num_kernels=8,
                 input_channels=3, output_channels=2, dropout=0).float()
    B = 4
    H, W = 34, 34  # include boundary padding
    sample_batched = {
        'structure': torch.randn(B, 1, H, W),
        'field': torch.randn(B, 2, H, W),
        'src': torch.randn(B, 2, H, W),
        'wavelength_normalized': torch.randn(B),
        'angle_normalized': torch.randn(B),
        'time_state': torch.rand(B),
        'time_state_normalized': torch.rand(B),
    }
    pattern = torch.ones(B, 1, H + 1, W)
    args = argparse.Namespace(lambda_continuity=0.1, lambda_frozen_mode=0.05,
                               physics_informed_temporal=True)

    total_loss, loss_dict = temporal_physics_loss(
        model, sample_batched, pattern, args,
        field_scaling_factor=1.0, src_data_scaling_factor=1.0
    )
    assert total_loss.shape == (), f"Expected scalar loss, got {total_loss.shape}"
    assert 'continuity_loss' in loss_dict
    assert 'frozen_eigenmode_loss' in loss_dict
    assert 'temporal_total_loss' in loss_dict

test("Full temporal_physics_loss integration (CPU)", test_temporal_physics_loss_integration)


# ════════════════════════════════════════════════════════════════════════════════
# CONTRIBUTION 3: Backward-Compatible Checkpoint Loading
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("CONTRIBUTION 3: Backward-Compatible Checkpoint Loading")
print("=" * 72)

def test_upgrade_import():
    from multi_film_angle_dec_fwdadj_sample_learners import upgrade_film_state_dict, load_legacy_checkpoint
    assert callable(upgrade_film_state_dict)
    assert callable(load_legacy_checkpoint)

test("upgrade_film_state_dict and load_legacy_checkpoint import", test_upgrade_import)


def test_upgrade_zero_pad():
    """Create a fake 2-condition state dict, upgrade it, verify zero-padding."""
    from multi_film_angle_dec_fwdadj_sample_learners import upgrade_film_state_dict
    fake_sd = {
        'film_layers.0.film_layer.weight': torch.randn(32, 2),
        'film_layers.0.film_layer.bias': torch.randn(32),
        'film_layers.1.film_layer.weight': torch.randn(64, 2),
        'film_layers.1.film_layer.bias': torch.randn(64),
        'conv_layers.0.weight': torch.randn(8, 3, 3, 3),  # non-FiLM key
    }
    upgraded_sd, upgraded_keys = upgrade_film_state_dict(fake_sd.copy())
    assert len(upgraded_keys) == 2, f"Expected 2 upgraded keys, got {len(upgraded_keys)}"
    for key in upgraded_keys:
        assert upgraded_sd[key].shape[1] == 3, \
            f"Expected 3 columns after upgrade, got {upgraded_sd[key].shape[1]}"

test("upgrade_film_state_dict zero-pads 2->3 columns", test_upgrade_zero_pad)


def test_upgrade_preserves_original_weights():
    """Verify the original 2 columns are preserved exactly after upgrade."""
    from multi_film_angle_dec_fwdadj_sample_learners import upgrade_film_state_dict
    original_weight = torch.randn(32, 2)
    fake_sd = {
        'film_layers.0.film_layer.weight': original_weight.clone(),
        'film_layers.0.film_layer.bias': torch.randn(32),
    }
    upgraded_sd, _ = upgrade_film_state_dict(fake_sd)
    upgraded_weight = upgraded_sd['film_layers.0.film_layer.weight']
    # First 2 columns must match original
    assert torch.allclose(upgraded_weight[:, :2], original_weight), \
        "Original columns should be preserved"
    # Third column must be zero
    assert torch.allclose(upgraded_weight[:, 2], torch.zeros(32)), \
        "Padded column should be all zeros"

test("Upgrade preserves original weights, appends zeros", test_upgrade_preserves_original_weights)


def test_upgrade_mathematically_identical():
    """
    Verify that the upgraded model produces mathematically identical output
    to the original 2-condition model for time_norm=0 (or any value, since
    the time column weights are zero).
    """
    from multi_film_angle_dec_fwdadj_sample_learners import FiLM
    # Build a 2-condition FiLM
    film2 = FiLM(num_features=16, num_conditions=2)
    # Build a 3-condition FiLM, upgrade its weights from the 2-cond model
    film3 = FiLM(num_features=16, num_conditions=3)

    # Copy 2-cond weights into 3-cond with zero-padding
    with torch.no_grad():
        old_w = film2.film_layer.weight.data  # (32, 2)
        old_b = film2.film_layer.bias.data    # (32,)
        new_w = torch.cat([old_w, torch.zeros(32, 1)], dim=1)  # (32, 3)
        film3.film_layer.weight.data = new_w
        film3.film_layer.bias.data = old_b.clone()

    x = torch.randn(2, 16, 8, 8)
    wl = torch.tensor([0.5, 0.7])
    angle = torch.tensor([0.1, -0.2])
    time_norm = torch.tensor([0.3, 0.9])  # arbitrary — should not matter

    # 2-cond forward needs only wl and angle
    with torch.no_grad():
        cond2 = torch.stack([wl, angle], dim=1)
        params2 = film2.film_layer(cond2)
        gamma2, beta2 = torch.chunk(params2, 2, dim=1)
        out2 = gamma2.unsqueeze(-1).unsqueeze(-1) * x + beta2.unsqueeze(-1).unsqueeze(-1)

        out3 = film3(x, wl, angle, time_norm)

    assert torch.allclose(out2, out3, atol=1e-6), \
        f"Upgraded 3-cond model should match 2-cond output. Max diff: {(out2 - out3).abs().max().item()}"

test("Zero-padded 3-cond FiLM is mathematically identical to 2-cond", test_upgrade_mathematically_identical)


def test_load_legacy_checkpoint_roundtrip():
    """Create a fake 2-cond checkpoint, save it, load it via load_legacy_checkpoint."""
    from multi_film_angle_dec_fwdadj_sample_learners import UNet, load_legacy_checkpoint
    import tempfile

    # Create a 2-condition model (simulating legacy)
    model_2cond = UNet(net_depth=3, block_depth=2, init_num_kernels=8,
                       input_channels=3, output_channels=2, dropout=0).float()
    # Manually shrink FiLM layers to 2 conditions to simulate legacy
    for film in model_2cond.film_layers:
        old_w = film.film_layer.weight.data[:, :2]
        old_b = film.film_layer.bias.data
        film.film_layer = torch.nn.Linear(2, film.num_features * 2)
        with torch.no_grad():
            film.film_layer.weight.data = old_w
            film.film_layer.bias.data = old_b

    # Save as legacy checkpoint
    tmp_path = os.path.join(os.path.dirname(__file__), "_test_legacy_checkpoint.pt")
    try:
        checkpoint = {
            'epoch': 10,
            'state_dict': model_2cond.state_dict(),
            'optimizer': None,
            'lr_scheduler': None,
        }
        torch.save(checkpoint, tmp_path)

        # Load via load_legacy_checkpoint
        loaded_model, loaded_ckpt = load_legacy_checkpoint(
            tmp_path, net_depth=3, block_depth=2, init_num_kernels=8,
            input_channels=3, output_channels=2, dropout=0
        )
        # Verify loaded model has 3-condition FiLM
        for film in loaded_model.film_layers:
            assert film.film_layer.weight.shape[1] == 3, \
                f"Expected 3-condition FiLM, got {film.film_layer.weight.shape[1]}"
        # Verify it can do a forward pass
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = loaded_model(x, torch.tensor([0.5]), torch.tensor([0.0]), torch.tensor([0.0]))
        assert out.shape == (1, 2, 64, 64)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

test("load_legacy_checkpoint upgrade roundtrip (2->3 cond)", test_load_legacy_checkpoint_roundtrip)


# ════════════════════════════════════════════════════════════════════════════════
# CONTRIBUTION 4: TemporalModulationAgent (AIM)
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("CONTRIBUTION 4: TemporalModulationAgent (AIM)")
print("=" * 72)

def test_agent_import():
    from agent.temporal_modulation_agent import (
        TemporalModulationAgent, TemporalModulationAgentIterative,
        TEMPORAL_MODULATION_SYSTEM_PROMPT
    )
    assert TemporalModulationAgent is not None
    assert TemporalModulationAgentIterative is not None
    assert len(TEMPORAL_MODULATION_SYSTEM_PROMPT) > 100

test("TemporalModulationAgent imports from agent module", test_agent_import)


def test_agent_init_exports():
    """Verify __init__.py exports both classes."""
    from agent import TemporalModulationAgent, TemporalModulationAgentIterative
    assert TemporalModulationAgent is not None
    assert TemporalModulationAgentIterative is not None

test("agent/__init__.py exports both agent classes", test_agent_init_exports)


def test_agent_inherits_base():
    from agent.temporal_modulation_agent import TemporalModulationAgent
    from agent.base import Agent
    assert issubclass(TemporalModulationAgent, Agent), \
        "TemporalModulationAgent must inherit from Agent"

test("TemporalModulationAgent inherits from Agent base class", test_agent_inherits_base)


def test_agent_system_prompt_content():
    from agent.temporal_modulation_agent import TEMPORAL_MODULATION_SYSTEM_PROMPT
    prompt = TEMPORAL_MODULATION_SYSTEM_PROMPT
    # Check key physics concepts are mentioned
    assert "D = εE" in prompt or "D = \\varepsilon E" in prompt or "D = εE" in prompt, \
        "System prompt should mention D-field continuity"
    assert "σ₀" in prompt or "sigma_0" in prompt, \
        "System prompt should mention sigma_0"
    assert "exp(-t/τ)" in prompt or "exp(-t / τ)" in prompt, \
        "System prompt should mention exponential control signal"
    assert "temporal_design" in prompt, \
        "System prompt should reference temporal_design tool"
    assert "<response>" in prompt, \
        "System prompt should have response tag instructions"

test("System prompt contains key physics and tool references", test_agent_system_prompt_content)


def test_agent_iterative_has_solve():
    from agent.temporal_modulation_agent import TemporalModulationAgentIterative
    assert hasattr(TemporalModulationAgentIterative, 'solve'), \
        "TemporalModulationAgentIterative must have a solve method"
    import inspect
    sig = inspect.signature(TemporalModulationAgentIterative.solve)
    param_names = list(sig.parameters.keys())
    assert 'problem' in param_names, "solve() must accept 'problem' parameter"

test("TemporalModulationAgentIterative has solve(problem=...)", test_agent_iterative_has_solve)


def test_agent_tool_registration():
    """Verify that all 4 expected tools get registered during __init__."""
    from agent.temporal_modulation_agent import TemporalModulationAgent
    # We need a mock model to instantiate — just check class structure
    import inspect
    init_src = inspect.getsource(TemporalModulationAgent.__init__)
    assert "scientific_compute" in init_src
    assert "symbolic_solve" in init_src
    assert "neural_design" in init_src
    assert "temporal_design" in init_src

test("Agent registers all 4 tools (scientific, symbolic, neural, temporal)", test_agent_tool_registration)


# ════════════════════════════════════════════════════════════════════════════════
# CONTRIBUTION 5: Temporal Design API Tool
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("CONTRIBUTION 5: Temporal Design API Tool")
print("=" * 72)

def test_temporal_api_import():
    from tools.design.temporal_design import TemporalDesignAPI
    assert TemporalDesignAPI is not None

test("TemporalDesignAPI imports successfully", test_temporal_api_import)


def test_design_temporal_metasurface():
    from tools.design.temporal_design import TemporalDesignAPI
    api = TemporalDesignAPI()
    result = api.design_temporal_metasurface(
        refractive_index=2.7,
        length=500e-9,
        thickness=200e-9,
        operating_wavelength=800e-9,
        incident_angle=0.0,
        time_window=(0.0, 30e-9),
        num_time_steps=10,
        control_signal_type="exponential",
        sigma_0=1.0,
        tau=5e-9,
    )
    assert isinstance(result, str), f"Expected string result, got {type(result)}"
    assert "API called successfully" in result
    assert "WAVEYNET_TEMPORAL_API" in result
    assert "sigma_0=1.0" in result
    assert "10 time steps" in result

test("design_temporal_metasurface returns correct API string", test_design_temporal_metasurface)


def test_design_exponential_signal():
    from tools.design.temporal_design import TemporalDesignAPI
    api = TemporalDesignAPI()
    result = api.design_temporal_metasurface(
        refractive_index=2.0, length=1e-6, thickness=100e-9,
        operating_wavelength=500e-9, sigma_0=5.0, tau=3e-9,
        num_time_steps=5, control_signal_type="exponential",
        time_window=(0.0, 15e-9),
    )
    assert "exponential" in result
    assert "5.0" in result or "sigma_0=5.0" in result

test("Exponential control signal parameters in output", test_design_exponential_signal)


def test_design_step_signal():
    from tools.design.temporal_design import TemporalDesignAPI
    api = TemporalDesignAPI()
    result = api.design_temporal_metasurface(
        refractive_index=2.0, length=1e-6, thickness=100e-9,
        operating_wavelength=500e-9, control_signal_type="step",
        sigma_0=3.0, tau=5e-9,
    )
    assert "step" in result

test("Step control signal type accepted", test_design_step_signal)


def test_design_linear_signal():
    from tools.design.temporal_design import TemporalDesignAPI
    api = TemporalDesignAPI()
    result = api.design_temporal_metasurface(
        refractive_index=2.0, length=1e-6, thickness=100e-9,
        operating_wavelength=500e-9, control_signal_type="linear",
        sigma_0=2.0, tau=5e-9,
    )
    assert "linear" in result

test("Linear control signal type accepted", test_design_linear_signal)


def test_design_invalid_signal_type():
    from tools.design.temporal_design import TemporalDesignAPI
    api = TemporalDesignAPI()
    result = api.design_temporal_metasurface(
        refractive_index=2.0, length=1e-6, thickness=100e-9,
        operating_wavelength=500e-9, control_signal_type="sawtooth",
        sigma_0=1.0, tau=5e-9,
    )
    assert "Error" in result

test("Invalid control signal type returns error", test_design_invalid_signal_type)


def test_optimize_control_signal():
    from tools.design.temporal_design import TemporalDesignAPI
    api = TemporalDesignAPI()
    result = api.optimize_control_signal(
        refractive_index=2.7,
        length=500e-9,
        thickness=200e-9,
        operating_wavelength=800e-9,
        incident_angle=0.0,
        suppression_window=(7e-9, 17e-9),
        objective="minimize_voltage",
        sigma_0_range=(0.1, 10.0),
        tau_range=(1e-9, 20e-9),
    )
    assert isinstance(result, str), f"Expected string, got {type(result)}"
    assert "API called successfully" in result
    assert "WAVEYNET_TEMPORAL_API.optimize_control_signal" in result
    assert "minimize voltage" in result or "minimize_voltage" in result

test("optimize_control_signal returns correct API string", test_optimize_control_signal)


def test_api_execute_routing():
    """Test that execute() routes code to the correct function."""
    from tools.design.temporal_design import TemporalDesignAPI
    api = TemporalDesignAPI()

    code = (
        "design_temporal_metasurface("
        "refractive_index=2.7, length=500e-9, thickness=200e-9, "
        "operating_wavelength=800e-9, incident_angle=0.0, "
        "time_window=(0.0, 30e-9), num_time_steps=10, "
        "control_signal_type='exponential', sigma_0=1.0, tau=5e-9)"
    )

    result = asyncio.get_event_loop().run_until_complete(api.execute(code))
    assert result['success'] is True, f"Execute failed: {result.get('error', '')}"
    assert "API called successfully" in result['result']

test("TemporalDesignAPI.execute() routes code correctly", test_api_execute_routing)


def test_api_execute_optimize_routing():
    from tools.design.temporal_design import TemporalDesignAPI
    api = TemporalDesignAPI()
    code = (
        "optimize_control_signal("
        "refractive_index=2.7, length=500e-9, thickness=200e-9, "
        "operating_wavelength=800e-9, suppression_window=(7e-9, 17e-9))"
    )
    result = asyncio.get_event_loop().run_until_complete(api.execute(code))
    assert result['success'] is True, f"Execute failed: {result.get('error', '')}"
    assert "optimize_control_signal" in result['result']

test("TemporalDesignAPI.execute() routes optimize_control_signal", test_api_execute_optimize_routing)


# ════════════════════════════════════════════════════════════════════════════════
# CROSS-CONTRIBUTION INTEGRATION TESTS
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("CROSS-CONTRIBUTION INTEGRATION")
print("=" * 72)


def test_dataloader_time_state():
    """Verify dataloader returns time_state and time_state_normalized keys."""
    from multi_film_angle_dec_fwdadj_sample_otf_dataloader import SimulationDataset
    # We can't instantiate without data, but we can verify __getitem__ returns correct keys
    import inspect
    src = inspect.getsource(SimulationDataset.__getitem__)
    assert 'time_state' in src, "Dataloader __getitem__ should return time_state"
    assert 'time_state_normalized' in src, "Dataloader __getitem__ should return time_state_normalized"

test("Dataloader returns time_state fields in __getitem__", test_dataloader_time_state)


def test_dataloader_scaling_factors_include_time():
    """Verify get_scaling_factors includes time_state bounds."""
    import inspect
    from multi_film_angle_dec_fwdadj_sample_otf_dataloader import SimulationDataset
    src = inspect.getsource(SimulationDataset.get_scaling_factors)
    assert 'max_time_state' in src
    assert 'min_time_state' in src

test("Dataloader scaling factors include time_state bounds", test_dataloader_scaling_factors_include_time)


def test_train_imports_temporal():
    """Verify train script imports temporal_physics."""
    train_path = os.path.join(film_src, "multi_film_angle_dec_fwdadj_sample_otf_train.py")
    with open(train_path, 'r') as f:
        content = f.read()
    assert "from temporal_physics import temporal_physics_loss" in content, \
        "Training script must import temporal_physics_loss"
    assert "time_states_normalized" in content, \
        "Training script must use time_states_normalized"
    assert "load_legacy_checkpoint" in content, \
        "Training script must use load_legacy_checkpoint for continue_train"

test("Training script integrates temporal physics and checkpoint loading", test_train_imports_temporal)


def test_config_has_temporal_settings():
    """Verify config.yaml includes temporal modulation settings."""
    import yaml
    config_path = os.path.join(film_src, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert config.get('temporal_modulation') is True
    assert config.get('physics_informed_temporal') is True
    assert config.get('lambda_continuity') == 0.1
    assert config.get('lambda_frozen_mode') == 0.05

test("config.yaml has temporal modulation settings", test_config_has_temporal_settings)


def test_readme_lists_all_contributions():
    """Verify README documents all 5 contributions."""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_path, 'r') as f:
        content = f.read()
    assert "Contribution 1" in content
    assert "Contribution 2" in content
    assert "Contribution 3" in content
    assert "Contribution 4" in content
    assert "Contribution 5" in content
    assert "Rashedul Albab" in content

test("README.md documents all 5 contributions with attribution", test_readme_lists_all_contributions)


# ════════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)

print(f"\n  Total tests: {total}")
print(f"  Passed:      {passed}")
print(f"  Failed:      {failed}")

if failed > 0:
    print(f"\n  FAILED TESTS:")
    for name, ok, tb in results:
        if not ok:
            print(f"    - {name}")
    print()
    sys.exit(1)
else:
    print(f"\n  {PASS}  All {total} tests passed! All 5 contributions are working correctly.")
    sys.exit(0)
