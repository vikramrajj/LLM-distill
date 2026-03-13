# LLM-distill 🧪

**Reinforcement Distillation for Local Agentic LLMs**

Build lightweight, agentic AI systems that learn from their environment. Run powerful agents on consumer hardware using distilled models with tool-use capabilities.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Agent Loop                      │
│                                                  │
│  ┌─────────────┐    ┌────────────────────────┐  │
│  │  Local LLM  │◀──▶│  Tool Router           │  │
│  │  (Phi-3.5)  │    │  ├── shell_exec        │  │
│  │  llama.cpp  │    │  ├── browser (Playwright)│ │
│  └─────────────┘    │  ├── file_ops          │  │
│        ▲            │  ├── web_fetch         │  │
│        │            │  └── python_exec       │  │
│  ┌─────┴──────┐     └────────────────────────┘  │
│  │ Experience │              │                   │
│  │  Buffer    │◀─────────────┘                   │
│  └─────┬──────┘                                  │
│        │                                         │
│  ┌─────▼──────────┐                              │
│  │  RL / GRPO     │  (Distillation Pipeline)    │
│  │  Training Loop │                              │
│  └────────────────┘                              │
└─────────────────────────────────────────────────┘
```

## Components

- **`agent/`** — Core agent loop with ReAct-style reasoning
- **`tools/`** — Tool implementations (shell, browser, file, web)
- **`llm/`** — LLM interface (OpenAI-compatible, works with llama.cpp)
- **`training/`** — RL/GRPO distillation pipeline
- **`experiments/`** — Training configs and experiment tracking

## Quick Start

```bash
# 1. Start the local LLM server
./scripts/start_llm.sh

# 2. Run the agent
python -m agent.cli "List the files in the current directory and summarize what you find"

# 3. Run with browser tools
python -m agent.cli --browser "Search for 'local LLM agents' and summarize the top 3 results"
```

## Requirements

- Python 3.10+
- llama.cpp with a GGUF model (Phi-3.5-mini recommended)
- Playwright (for browser automation)

## Model

Default: **Phi-3.5-mini-instruct** (Q4_K_M, 3.8B params, ~2.3GB)
- Runs on CPU with AVX-512 at ~10 tok/s
- Fits in ~6GB RAM with 4K context
- OpenAI-compatible API via llama.cpp server

## License

MIT
