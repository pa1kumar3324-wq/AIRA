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
from cli.interface import CLIInterface


def main():
    parser = argparse.ArgumentParser(description="AIRA — Emotionally Intelligent AI Chatbot")
    parser.add_argument("--user", type=str, default="User", help="Your display name")
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
