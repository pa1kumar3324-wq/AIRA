"""
AIRA — Emotionally Intelligent AI Chatbot
Entry point. Run this to start the CLI chat interface.

Usage:
    python main.py
    python main.py --user YourName
    python main.py --history          # show past sessions
    python main.py --clear            # clear all memory
"""

# ============================================================
# HARDWARE-AWARE MODEL BOOTSTRAP  (runs before anything else)
# ============================================================
# 1. Profile the host hardware (RAM / CPU / GPU VRAM).
# 2. Pick the best Ollama model the machine can support.
# 3. Verify the model is pulled locally; pull it if not.
# 4. Inject the result into config.json so the rest of the
#    application (LLMClient, etc.) picks it up transparently.
# On any failure the code falls back to the Tier-3 model so
# the application always starts cleanly.

try:
    import ollama as _ollama                    # pip install ollama
    import colorama as _colorama
    from colorama import Fore as _Fore, Style as _Style
    _colorama.init(autoreset=True)

    from hardware_profiler import (
        profile_hardware  as _profile_hw,
        select_model      as _select_model,
        TIER3_MODEL       as _FALLBACK_MODEL,
        TIER1_MODEL, TIER2_MODEL, TIER3_MODEL,
    )

    # ── Colour aliases (local to bootstrap) ──────────────────────────────────
    _C  = _Fore.CYAN
    _G  = _Fore.GREEN
    _Y  = _Fore.YELLOW
    _R  = _Fore.RED
    _M  = _Fore.MAGENTA
    _W  = _Fore.WHITE
    _B  = _Style.BRIGHT
    _D  = "\033[2m"          # dim
    _RS = _Style.RESET_ALL

    def _bootstrap_model() -> str:
        """Profile hardware, select, verify and (if needed) pull the best Ollama model."""

        # ── 1. Profile hardware ───────────────────────────────────────────────
        try:
            _hw  = _profile_hw()
            chosen = _select_model(_hw)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "[Bootstrap] Hardware profiling failed (%s). Using fallback: %s",
                exc, _FALLBACK_MODEL,
            )
            _hw    = None
            chosen = _FALLBACK_MODEL

        # ── 2. Determine tier label & colour ──────────────────────────────────
        if chosen == TIER1_MODEL:
            _tier_label = "Tier 1  (High-end)"
            _tier_color = _G
        elif chosen == TIER2_MODEL:
            _tier_label = "Tier 2  (Mid-range)"
            _tier_color = _C
        else:
            _tier_label = "Tier 3  (Lightweight)"
            _tier_color = _Y

        # ── 3. Print startup card ─────────────────────────────────────────────
        _w = 47
        _bar = _D + "-" * _w + _RS
        print()
        print(_C + _B + "  " + "-" * _w + _RS)
        print(_C + _B + "  |{:^{w}}|".format(" AIRA  Hardware Detector ", w=_w - 2) + _RS)
        print(_C + _B + "  " + "-" * _w + _RS)
        print()

        if _hw:
            _ram_str  = "{:.1f} GB".format(_hw.total_ram_gb)
            _cpu_str  = "{} logical cores".format(_hw.cpu_cores)
            _gpu_str  = (_hw.gpu_name + "  ({:.1f} GB VRAM)".format(_hw.gpu_vram_gb)
                         if _hw.gpu_detected else "Not detected")

            print("  {}{}RAM{}{:<26}{}{}".format(_B, _W, _RS, "", _C, _ram_str + _RS))
            print("  {}{}CPU{}{:<26}{}{}".format(_B, _W, _RS, "", _C, _cpu_str + _RS))
            print("  {}{}GPU{}{:<26}{}{}".format(_B, _W, _RS, "", _C, _gpu_str + _RS))
        else:
            print("  {}{}Hardware profile unavailable{}".format(_D, _Y, _RS))

        print()
        print(_bar)
        print()
        print("  {}{} Tier{}{:<22}{}{}{}".format(
            _B, _W, _RS, "", _tier_color, _B, _tier_label + _RS))
        print("  {}{}Model{}{:<21}{}{}{}".format(
            _B, _W, _RS, "", _G, _B, chosen + _RS))
        print()
        print(_C + _B + "  " + "-" * _w + _RS)
        print()

        # ── 4. Verify / pull ──────────────────────────────────────────────────
        try:
            response   = _ollama.list()
            raw_models = getattr(response, "models", None) or response.get("models", [])
            installed_tags = [
                getattr(m, "model", None) or m.get("name", "") or ""
                for m in raw_models
            ]
            installed_base = [tag.split(":")[0].lower() for tag in installed_tags]
            chosen_base    = chosen.split(":")[0].lower()

            if chosen_base not in installed_base:
                print("  {}{}[!]{} Model '{}{}{}' not found locally.".format(
                    _B, _Y, _RS, _B, chosen, _RS))
                print("  {}    Pulling from Ollama registry - this may take a while...{}".format(_D, _RS))
                print()
                _ollama.pull(chosen)
                print()
                print("  {}{}[OK]{} Model '{}{}{}' pulled successfully.".format(
                    _B, _G, _RS, _B, chosen, _RS))
            else:
                print("  {}{}[OK]{} Model '{}{}{}' is ready.".format(
                    _B, _G, _RS, _B, chosen, _RS))
            print()

        except Exception as exc:
            import logging
            logging.getLogger(__name__).error(
                "[Bootstrap] Ollama verify/pull failed (%s). Proceeding with '%s'.",
                exc, chosen,
            )
            print("  {}{}[!]{} Could not verify model - proceeding anyway.".format(_B, _Y, _RS))
            print()

        return chosen

    _SELECTED_MODEL = _bootstrap_model()

except ImportError as _import_err:
    # ollama or psutil not installed — degrade gracefully
    import logging as _logging
    _logging.basicConfig()
    _logging.getLogger(__name__).warning(
        "[Bootstrap] Could not import required library (%s). "
        "Install with: pip install ollama psutil  — defaulting to 'phi3'.",
        _import_err,
    )
    _SELECTED_MODEL = "phi3"

# ─── Patch the live config so LLMClient sees the selected model ───────────
import json as _json
import pathlib as _pathlib
_cfg_path = _pathlib.Path(__file__).parent / "config.json"
try:
    with open(_cfg_path, "r") as _f:
        _cfg = _json.load(_f)
    _cfg["model"] = _SELECTED_MODEL
    with open(_cfg_path, "w") as _f:
        _json.dump(_cfg, _f, indent=2)
except Exception:
    pass   # Non-fatal; worst case the old model name in config is used
# ============================================================
# END HARDWARE BOOTSTRAP
# ============================================================

import argparse
import json
import logging
from pathlib import Path
from cli.interface import CLIInterface

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_FILE = Path(__file__).parent / "config.json"

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {"username": "User"}

def main():
    config = load_config()
    default_user = config.get("username", "User")

    parser = argparse.ArgumentParser(description="AIRA — Emotionally Intelligent AI Chatbot")
    parser.add_argument("--user", type=str, default=default_user, help="Your display name")
    parser.add_argument("--history", action="store_true", help="Show past conversation sessions")
    parser.add_argument("--clear", action="store_true", help="Clear all saved conversation history")
    args = parser.parse_args()

    cli = CLIInterface(username=args.user)

    if args.history:
        cli.show_history()
        return

    if args.clear:
        cli.clear_history()
        return

    cli.start()


if __name__ == "__main__":
    main()
