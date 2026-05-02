"""
AIRA — Emotionally Intelligent AI Chatbot
Entry point. Run this to start the CLI chat interface.

Usage:
    python main.py
    python main.py --user YourName
    python main.py --history          # show past sessions
    python main.py --clear            # clear all memory
"""

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
