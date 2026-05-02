import logging
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import psutil  # pip install psutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier thresholds (in gigabytes)
# ---------------------------------------------------------------------------
TIER1_RAM_GB        = 32   # RAM floor for Tier 1 when no GPU data available
TIER2_RAM_GB_MIN    = 16   # Lower bound of Tier 2 RAM window
TIER1_VRAM_GB       = 16   # GPU VRAM that alone qualifies as Tier 1

# ---------------------------------------------------------------------------
# Model names per tier
# ---------------------------------------------------------------------------
TIER1_MODEL  = "llama3:70b"
TIER2_MODEL  = "llama3:8b"
TIER3_MODEL  = "phi3"
TIER3_FALLBACK_MODEL = "qwen2:0.5b"  # ultimate fallback if phi3 unavailable


# ---------------------------------------------------------------------------
# Hardware snapshot dataclass
# ---------------------------------------------------------------------------
@dataclass
class HardwareProfile:
    """Holds raw hardware metrics collected from the host system."""
    total_ram_gb: float = 0.0
    cpu_cores: int = 0
    gpu_name: Optional[str] = None
    gpu_vram_gb: float = 0.0
    gpu_detected: bool = False
    profiling_errors: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_ram_gb() -> float:
    """Return total system RAM in gigabytes (float, 2 d.p.)."""
    try:
        ram_bytes = psutil.virtual_memory().total
        return round(ram_bytes / (1024 ** 3), 2)
    except Exception as exc:
        logger.warning("RAM detection failed: %s", exc)
        return 0.0


def _get_cpu_cores() -> int:
    """Return the number of logical CPU cores."""
    try:
        return psutil.cpu_count(logical=True) or 1
    except Exception as exc:
        logger.warning("CPU core detection failed: %s", exc)
        return 1


def _get_nvidia_vram_gb() -> tuple[Optional[str], float]:
    """
    Query NVIDIA GPU VRAM via nvidia-smi.

    Returns:
        (gpu_name, vram_gb) — gpu_name is None and vram_gb is 0.0 if not found.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Take the first GPU line only
            first_line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            gpu_name = parts[0]
            vram_mb  = float(parts[1])
            return gpu_name, round(vram_mb / 1024, 2)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass  # nvidia-smi not available — normal on AMD / CPU-only machines
    except Exception as exc:
        logger.debug("nvidia-smi query error: %s", exc)
    return None, 0.0


def _get_amd_vram_gb() -> tuple[Optional[str], float]:
    """
    Attempt to detect AMD GPU VRAM.
    On Windows we query wmic; on Linux we check /sys/class/drm.

    Returns:
        (gpu_name, vram_gb) — gpu_name is None and vram_gb is 0.0 if not found.
    """
    system = platform.system()

    if system == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController",
                 "get", "name,AdapterRAM", "/format:csv"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        adapter_ram_str, gpu_name = parts[1].strip(), parts[2].strip()
                        if adapter_ram_str.isdigit() and int(adapter_ram_str) > 0:
                            vram_gb = round(int(adapter_ram_str) / (1024 ** 3), 2)
                            return gpu_name, vram_gb
        except Exception as exc:
            logger.debug("WMIC AMD query error: %s", exc)

    elif system == "Linux":
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                # rocm-smi JSON structure varies; best-effort extraction
                for card, info in data.items():
                    vram_total = info.get("VRAM Total Memory (B)", 0)
                    if vram_total:
                        return card, round(int(vram_total) / (1024 ** 3), 2)
        except Exception as exc:
            logger.debug("rocm-smi query error: %s", exc)

    return None, 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_hardware() -> HardwareProfile:
    """
    Collect hardware metrics and return a populated HardwareProfile.

    This function never raises — any detection error is recorded inside
    HardwareProfile.profiling_errors so the caller can still proceed.
    """
    hw = HardwareProfile()

    # --- RAM ---
    hw.total_ram_gb = _get_ram_gb()

    # --- CPU ---
    hw.cpu_cores = _get_cpu_cores()

    # --- GPU (NVIDIA first, then AMD fallback) ---
    try:
        gpu_name, vram_gb = _get_nvidia_vram_gb()
        if gpu_name is None:
            gpu_name, vram_gb = _get_amd_vram_gb()

        if gpu_name:
            hw.gpu_name      = gpu_name
            hw.gpu_vram_gb   = vram_gb
            hw.gpu_detected  = True
    except Exception as exc:
        hw.profiling_errors.append(f"GPU detection error: {exc}")

    return hw


def select_model(hw: Optional[HardwareProfile] = None) -> str:
    """
    Apply the tiered decision logic and return the recommended Ollama model tag.

    Decision tree
    ─────────────
    Tier 1  GPU VRAM ≥ TIER1_VRAM_GB GB   OR   RAM > TIER1_RAM_GB GB
    Tier 2  RAM is between TIER2_RAM_GB_MIN and TIER1_RAM_GB (inclusive)
    Tier 3  Everything else (low RAM / no GPU)

    Args:
        hw: A HardwareProfile instance. If None, profile_hardware() is called.

    Returns:
        str — Ollama model tag (e.g. "llama3:8b")
    """
    if hw is None:
        hw = profile_hardware()

    logger.info(
        "[HardwareProfiler] RAM=%.1f GB | CPU cores=%d | GPU=%s (VRAM=%.1f GB)",
        hw.total_ram_gb,
        hw.cpu_cores,
        hw.gpu_name or "None detected",
        hw.gpu_vram_gb,
    )

    # --- Tier 1 ---
    if hw.gpu_vram_gb >= TIER1_VRAM_GB or hw.total_ram_gb > TIER1_RAM_GB:
        chosen = TIER1_MODEL
        tier   = 1

    # --- Tier 2 ---
    elif TIER2_RAM_GB_MIN <= hw.total_ram_gb <= TIER1_RAM_GB:
        chosen = TIER2_MODEL
        tier   = 2

    # --- Tier 3 ---
    else:
        chosen = TIER3_MODEL
        tier   = 3

    logger.info("[HardwareProfiler] Selected Tier %d model → %s", tier, chosen)
    return chosen


def get_recommended_model() -> str:
    """
    Convenience one-liner: profile hardware and return the recommended model.
    Equivalent to: select_model(profile_hardware())
    """
    return select_model(profile_hardware())
