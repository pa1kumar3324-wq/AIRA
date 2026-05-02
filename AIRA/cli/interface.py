"""
cli/interface.py

CLI chat interface for AIRA.
Handles the main conversation loop, display formatting, and user commands.

Commands (type during chat):
    /quit or /exit      — end the session and save history
    /emotion            — show the detected emotion for your last message
    /history            — show past session summaries
    /clear              — clear current session memory
    /help               — show available commands
"""

import sys
import os
import logging
from datetime import datetime

import colorama
from colorama import Fore, Back, Style

from core.sentiment import SentimentEngine
from core.llm import LLMClient
from memory.conversation import ConversationMemory

# Initialize colorama for cross-platform color support
colorama.init(autoreset=True)

# ── ANSI colour codes ────────────────────────────────────────────────────────
RESET   = Style.RESET_ALL
BOLD    = Style.BRIGHT
DIM     = "\033[2m"  # Dim not directly in colorama, keep ANSI
CYAN    = Fore.CYAN
GREEN   = Fore.GREEN
YELLOW  = Fore.YELLOW
RED     = Fore.RED
MAGENTA = Fore.MAGENTA
WHITE   = Fore.WHITE
BLUE    = Fore.BLUE

COMMANDS = {
    "/quit":    "End the session and save history",
    "/exit":    "End the session and save history",
    "/emotion": "Show the detected emotion for your last message",
    "/history": "Show past session summaries",
    "/clear":   "Clear current session memory",
    "/help":    "Show this help message",
}


class CLIInterface:
    def __init__(self, username: str = "User") -> None:
        self.username  = username
        self.sentiment = SentimentEngine()
        self.llm       = LLMClient()
        self.memory    = ConversationMemory(username=username)
        # Reflect the model that was selected by the hardware profiler at boot
        self._active_model = self.llm.model

    # ── Session lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        logging.info("Starting AIRA CLI interface")
        self._print_banner()
        self._check_ollama()

        try:
            self._chat_loop()
        except KeyboardInterrupt:
            logging.info("Session interrupted by user")
            self._end_session()

    def _chat_loop(self) -> None:
        while True:
            user_input = self._prompt_user()

            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                should_exit = self._handle_command(user_input.strip().lower())
                if should_exit:
                    break
                continue

            # Analyse sentiment
            sentiment = self.sentiment.analyse(user_input)
            emotion   = sentiment["emotion"]
            tone_hint = sentiment["tone_hint"]

            # Show detected emotion indicator
            label = self.sentiment.display_label(emotion)
            print(f"  {DIM}{label}{RESET}")
            print()

            # Store user message
            self.memory.add_user_message(user_input, emotion=emotion)

            # Get LLM response (streaming)
            print(f"{CYAN}{BOLD}AIRA{RESET}  ", end="", flush=True)
            full_response = ""
            for token in self.llm.chat(self.memory.get_context(), tone_hint=tone_hint):
                print(token, end="", flush=True)
                full_response += token
            print("\n")

            # Store assistant response
            self.memory.add_assistant_message(full_response)

    def _end_session(self) -> None:
        print(f"\n\n{DIM}Saving session...{RESET}")
        self.memory.save_session()
        msg_count = self.memory.message_count()
        print(f"{GREEN}Session saved. {msg_count} messages logged.{RESET}")
        print(f"{DIM}Goodbye, {self.username}!{RESET}\n")
        sys.exit(0)

    # ── Commands ─────────────────────────────────────────────────────────────

    def _handle_command(self, cmd: str) -> bool:
        """Handle a slash command. Returns True if the app should exit."""
        if cmd in ("/quit", "/exit"):
            self._end_session()
            return True

        elif cmd == "/emotion":
            emotion = self.memory.last_emotion()
            if emotion:
                label = self.sentiment.display_label(emotion)
                print(f"  Last detected emotion: {YELLOW}{label}{RESET}\n")
            else:
                print(f"  {DIM}No emotion detected yet.{RESET}\n")

        elif cmd == "/history":
            self.show_history()

        elif cmd == "/clear":
            self.memory.clear_session()
            print(f"  {GREEN}Session memory cleared.{RESET}\n")

        elif cmd == "/help":
            self._print_help()

        else:
            print(f"  {RED}Unknown command: {cmd}{RESET}  (type /help for options)\n")

        return False

    # ── Display ──────────────────────────────────────────────────────────────

    def _prompt_user(self) -> str:
        try:
            return input(f"{WHITE}{BOLD}{self.username}{RESET}  ")
        except EOFError:
            return "/quit"

    def _print_banner(self):
        os.system("clear" if os.name == "posix" else "cls")
        model_line = "  Powered by " + self._active_model + "  via Ollama"
        _w = 47
        print()
        print(CYAN + BOLD + "  " + "=" * _w + RESET)
        print(CYAN + BOLD + "  |{:^{w}}|".format("A I R A  --  Emotionally Intelligent AI", w=_w - 2) + RESET)
        print(CYAN + BOLD + "  |{:^{w}}|".format(model_line, w=_w - 2) + RESET)
        print(CYAN + BOLD + "  " + "=" * _w + RESET)
        print()
        print("  Welcome, {}{}{}. Type {}{}{} for commands.\n".format(
            BOLD, self.username, RESET, DIM, "/help", RESET))
        print("  {}{}{}\n".format(DIM, "-" * 45, RESET))


    def _print_help(self):
        print(f"\n  {BOLD}Available commands:{RESET}")
        for cmd, desc in COMMANDS.items():
            print(f"    {YELLOW}{cmd:<12}{RESET} {DIM}{desc}{RESET}")
        print()

    def _check_ollama(self):
        model_display = self._active_model
        if not self.llm.is_available():
            print("  {}Warning:{} Ollama is not running.".format(YELLOW, RESET))
            print("  Start it with: {}ollama serve{}".format(DIM, RESET))
            print("  Then pull the model: {}ollama pull {}{}".format(DIM, model_display, RESET))
            print()
        else:
            models = self.llm.list_models()
            model_base = model_display.split(":")[0].lower()
            if not any(model_base in m.lower() for m in models):
                print("  {}Warning:{} Model '{}{}{}' not found locally.".format(
                    YELLOW, RESET, BOLD, model_display, RESET))
                print("  Pull it with: {}ollama pull {}{}".format(DIM, model_display, RESET))
                print()
            else:
                print("  {}[OK]{} Ollama connected.  Model: {}{}{}{}".format(
                    GREEN, RESET, BOLD, CYAN, model_display, RESET))
                print()

    # ── History display ───────────────────────────────────────────────────────

    def show_history(self):
        sessions = self.memory.get_all_sessions()
        if not sessions:
            print(f"\n  {DIM}No past sessions found.{RESET}\n")
            return

        print(f"\n  {BOLD}Past sessions:{RESET}")
        print(f"  {DIM}{'─' * 55}{RESET}")
        for s in sessions[:10]:
            dt = datetime.fromisoformat(s["started_at"]).strftime("%d %b %Y  %H:%M")
            print(
                f"  {DIM}{s['session_id']}{RESET}  "
                f"{CYAN}{dt}{RESET}  "
                f"{s['message_count']} messages  "
                f"{DIM}({s['username']}){RESET}"
            )
        print()

    def clear_history(self):
        self.memory.clear_all_history()
        print(f"\n  {GREEN}All history cleared.{RESET}\n")
