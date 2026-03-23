# AIRA — Emotionally Intelligent AI Chatbot

> An AI chatbot that reads how you're feeling and responds accordingly.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Mistral-black?style=flat)
![VADER](https://img.shields.io/badge/Sentiment-VADER-orange?style=flat)
![Status](https://img.shields.io/badge/Status-Under%20Development-yellow?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## What is AIRA?

Most chatbots respond to *what* you say. AIRA responds to *how* you're feeling.

AIRA (Adaptive Intelligent Responsive Agent) analyses the emotional tone of every message you send using VADER sentiment analysis, then dynamically adjusts its personality and response style before passing the conversation to a locally running Mistral LLM via Ollama.

If you're happy, AIRA is upbeat. If you're stressed, it's calm and reassuring. If you're frustrated, it skips the fluff and gets to the point.

---

## Demo

```
AIRA  Welcome, Pavan. Type /help for commands.

Pavan   I have an exam tomorrow and I don't know where to start
        😟 anxious

AIRA    That's a tough spot to be in, but you've got more time than
        it feels like right now. Let's break it down — what subject
        is the exam on, and what topics feel shakiest to you?

Pavan   I actually just finished studying and feel great!
        😊 joyful

AIRA    That's awesome — seriously, that feeling after a solid study
        session is the best! You're going into that exam prepared.
        Go get some rest and trust the work you put in. You've got this!
```

---

## Features

- **Real-time sentiment analysis** — every message is scored and mapped to one of 7 emotional states
- **Tone-adaptive responses** — Mistral's system prompt is rewritten on every turn based on your mood
- **Streaming output** — responses appear token by token, just like ChatGPT
- **Persistent memory** — full conversation history saved to disk as JSON, browsable between sessions
- **Clean CLI interface** — colour-coded terminal UI with slash commands
- **Fully local** — no API keys, no cloud, everything runs on your machine via Ollama

---

## Emotional States

AIRA detects 7 emotional states and adapts its tone for each:

| Emotion | Trigger | How AIRA responds |
|---|---|---|
| 😊 Joy | Strong positive sentiment | Warm, enthusiastic, matches your energy |
| 🙂 Content | Mild positive sentiment | Friendly, encouraging |
| 😐 Neutral | Balanced or factual input | Clear, helpful, conversational |
| 😟 Anxious | Negative + question patterns | Calm, patient, reassuring |
| 😢 Sad | Negative, low energy | Gentle, empathetic, unhurried |
| 😠 Angry | Strongly negative | Composed, acknowledges feelings first |
| 😤 Frustrated | Terse or mildly negative | Direct, no fluff, solution-focused |

---

## How It Works

```
Your message
     │
     ▼
VADER Sentiment Analysis
     │  compound score + pattern matching
     ▼
Emotion Mapping
     │  joy / content / neutral / anxious / sad / angry / frustrated
     ▼
Tone Instruction Injection
     │  rewrites Mistral's system prompt for this turn
     ▼
Mistral LLM (via Ollama)
     │  streams response adapted to your emotional state
     ▼
Response + Memory Storage
```

---

## Project Structure

```
AIRA/
├── main.py                  # Entry point — run this to start AIRA
├── requirements.txt         # Python dependencies
├── README.md
│
├── core/
│   ├── sentiment.py         # VADER sentiment engine + emotion mapper
│   └── llm.py               # Ollama Mistral client with streaming support
│
├── memory/
│   ├── conversation.py      # In-session context + persistent session storage
│   └── sessions/            # Auto-created — stores JSON logs of each session
│
└── cli/
    └── interface.py         # Terminal chat UI, colour output, command handler
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com) installed and running

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/AIRA.git
cd AIRA
```

**2. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**3. Pull the Mistral model via Ollama**
```bash
ollama serve          # start the Ollama server
ollama pull mistral   # download Mistral (~4 GB)
```

**4. Run AIRA**
```bash
python main.py
```

---

## Usage

```bash
python main.py                    # start with default username
python main.py --user Pavan       # set your display name
python main.py --history          # view past session summaries
python main.py --clear            # delete all saved history
```

### In-chat commands

| Command | Description |
|---|---|
| `/help` | Show all available commands |
| `/emotion` | Show the emotion detected in your last message |
| `/history` | List past conversation sessions |
| `/clear` | Clear current session memory |
| `/quit` | Save session and exit |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Sentiment Analysis | [VADER](https://github.com/cjhutto/vaderSentiment) |
| Language Model | [Mistral](https://mistral.ai) via [Ollama](https://ollama.com) |
| Language | Python 3.10+ |
| Memory Storage | JSON (local filesystem) |
| Interface | Terminal CLI |

---

## Roadmap

- [x] VADER sentiment analysis engine
- [x] Ollama Mistral LLM integration with streaming
- [x] Emotion-to-tone mapping system
- [x] Persistent conversation memory
- [x] CLI interface with commands
- [ ] Web UI (Flask / FastAPI)
- [ ] Voice input via `speech_recognition`
- [ ] Session summarisation for long conversations
- [ ] Multi-model support (swap Mistral for other Ollama models)
- [ ] Emotion trend visualisation across sessions

---

## Contributing

This project is under active development. Contributions, suggestions, and feedback are welcome.

1. Fork the repository
2. Create a feature branch — `git checkout -b feature/your-feature`
3. Commit your changes — `git commit -m "add your feature"`
4. Push to the branch — `git push origin feature/your-feature`
5. Open a Pull Request

---

## Author

**Pavan Kumar M G**
B.E. Computer Science — RNS Institute of Technology, Bengaluru

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/pavan-kumar-m-g-4b59353b1)
[![Email](https://img.shields.io/badge/Email-pavankumar37979@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:pavankumar37979@gmail.com)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
