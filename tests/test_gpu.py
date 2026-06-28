"""GPU detection / high-performance-preference logic (no GL context needed)."""
from __future__ import annotations

from pycrossy import gpu


def test_detect_nvidia_shape():
    r = gpu.detect_nvidia()
    assert r is None or {"name", "driver", "vram_mb"} <= set(r)


def test_is_high_performance_classification():
    assert gpu.is_high_performance("NVIDIA GeForce RTX 3060 Laptop GPU/PCIe/SSE2")
    assert gpu.is_high_performance("AMD Radeon RX 6800 XT")
    assert not gpu.is_high_performance("Mesa Intel(R) UHD Graphics 620")
    assert not gpu.is_high_performance("AMD Radeon Graphics (radeonsi, renoir)")
    assert not gpu.is_high_performance("llvmpipe (LLVM 17.0)")


def test_prefer_can_opt_out(monkeypatch):
    monkeypatch.setenv("PYCROSSY_GPU", "integrated")
    result = gpu.prefer_high_performance_gpu(True)
    assert result["requested"] == "default"


def test_prefer_returns_request_dict():
    result = gpu.prefer_high_performance_gpu(enable=False)
    assert result["requested"] == "default"
    assert "nvidia" in result
