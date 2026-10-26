"""Tests for hardware inspection and fit estimation."""

from app.local.hardware import estimate_model_fit, get_hardware_info


def test_hardware_detection():
    hw = get_hardware_info()
    assert "ram" in hw
    assert hw["ram"]["total_gb"] > 0
    assert "cpu" in hw
    assert hw["cpu"]["cores"] > 0


def test_estimate_model_fit():
    # Mock hardware with 8 GB VRAM, 16 GB RAM
    mock_hw = {
        "ram": {"total_gb": 16.0, "available_gb": 12.0},
        "gpus": [{"name": "RTX 4070", "free_vram_gb": 8.0, "total_vram_gb": 12.0}],
    }

    # Small 4 GB model should fit fully on GPU
    fit_small = estimate_model_fit(4 * (1024 ** 3), mock_hw)
    assert fit_small["fit"] == "Full GPU"
    assert fit_small["suggested_ngl"] == 99

    # 12 GB model should fit in hybrid GPU + RAM
    fit_med = estimate_model_fit(12 * (1024 ** 3), mock_hw)
    assert fit_med["fit"] == "Hybrid (GPU + RAM)"

    # 40 GB model should exceed available memory
    fit_huge = estimate_model_fit(40 * (1024 ** 3), mock_hw)
    assert fit_huge["fit"] == "Insufficient RAM/VRAM"
