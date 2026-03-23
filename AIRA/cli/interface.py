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
from datetime import datetime

from core.sentiment import SentimentEngine
from core.llm import LLMClient
from memory.conversation import ConversationMemory

# ── ANSI colour codes ────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
MAGENTA = "\033[35m"
WHITE   = "\033[97m"
BLUE    = "\033[34m"

COMMANDS = {
    "/quit":    "End the session and save history",
    "/exit":    "End the session and save history",
    "/emotion": "Show the detected emotion for your last message",
    "/history": "Show past session summaries",
    "/clear":   "Clear current session memory",
    "/help":    "Show this help message",
}


class CLIInterface:
    def __init__(self, username: str = "User"):
        self.username  = username
        self.sentiment = SentimentEngine()
        self.llm       = LLMClient()
        self.memory    = ConversationMemory(username=username)

    # ── Session lifecycle ────────────────────────────────────────────────────

    def start(self):
        self._print_banner()
        self._check_ollama()

        try:
            self._chat_loop()
        except KeyboardInterrupt:
            self._end_session()

    def _chat_loop(self):
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

    def _end_session(self):
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
        print(f"""
{CYAN}{BOLD}
  ╔═══════════════════════════════════════════╗
  ║   A I R A  —  Emotionally Intelligent AI  ║
  ║   Powered by Mistral via Ollama           ║
  ╚═══════════════════════════════════════════╝
{RESET}""")
        print(f"  Welcome, {BOLD}{self.username}{RESET}. Type {DIM}/help{RESET} for commands.\n")
        print(f"  {DIM}{'─' * 45}{RESET}\n")

    def _print_help(self):
        print(f"\n  {BOLD}Available commands:{RESET}")
        for cmd, desc in COMMANDS.items():
            print(f"    {YELLOW}{cmd:<12}{RESET} {DIM}{desc}{RESET}")
        print()

    def _check_ollama(self):
        if not self.llm.is_available():
            print(f"  {YELLOW}Warning:{RESET} Ollama is not running.")
            print(f"  Start it with: {DIM}ollama serve{RESET}")
            print(f"  Then pull the model: {DIM}ollama pull mistral{RESET}\n")
        else:
            models = self.llm.list_models()
            if "mistral" not in " ".join(models):
                print(f"  {YELLOW}Warning:{RESET} Mistral model not found locally.")
                print(f"  Pull it with: {DIM}ollama pull mistral{RESET}\n")
            else:
                print(f"  {GREEN}Ollama connected. Mistral ready.{RESET}\n")

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
