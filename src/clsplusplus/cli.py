"""CLS++ CLI — Cross-model memory from your terminal."""

from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
from pathlib import Path

CONFIG_PATH = Path.home() / ".clsplusplus"
DEFAULT_URL = "https://www.clsplusplus.com"

# Model routing
OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini"}
ANTHROPIC_MODELS = {"claude", "claude-sonnet", "claude-haiku", "claude-opus",
                    "claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"}
GEMINI_MODELS = {"gemini", "gemini-pro", "gemini-2.0-flash"}


def _load_config() -> dict:
    """Load config from ~/.clsplusplus."""
    cfg = {
        "api_key": os.environ.get("CLS_API_KEY", ""),
        "user": os.environ.get("CLS_USER", os.environ.get("USER", "default")),
        "url": os.environ.get("CLS_BASE_URL", DEFAULT_URL),
        "openai_key": os.environ.get("OPENAI_API_KEY", ""),
        "anthropic_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "google_key": os.environ.get("GOOGLE_API_KEY", ""),
    }
    if CONFIG_PATH.exists():
        cp = configparser.ConfigParser()
        cp.read(CONFIG_PATH)
        if cp.has_section("default"):
            cfg["api_key"] = cp.get("default", "api_key", fallback=cfg["api_key"])
            cfg["user"] = cp.get("default", "user", fallback=cfg["user"])
            cfg["url"] = cp.get("default", "url", fallback=cfg["url"])
        if cp.has_section("keys"):
            cfg["openai_key"] = cp.get("keys", "openai", fallback=cfg["openai_key"])
            cfg["anthropic_key"] = cp.get("keys", "anthropic", fallback=cfg["anthropic_key"])
            cfg["google_key"] = cp.get("keys", "google", fallback=cfg["google_key"])
    return cfg


def _save_config(cfg: dict) -> None:
    """Save config to ~/.clsplusplus."""
    cp = configparser.ConfigParser()
    cp["default"] = {
        "api_key": cfg.get("api_key", ""),
        "user": cfg.get("user", ""),
        "url": cfg.get("url", DEFAULT_URL),
    }
    keys = {}
    if cfg.get("openai_key"):
        keys["openai"] = cfg["openai_key"]
    if cfg.get("anthropic_key"):
        keys["anthropic"] = cfg["anthropic_key"]
    if cfg.get("google_key"):
        keys["google"] = cfg["google_key"]
    if keys:
        cp["keys"] = keys
    with open(CONFIG_PATH, "w") as f:
        cp.write(f)
    print(f"Config saved to {CONFIG_PATH}")


def _brain(cfg: dict):
    """Create a Brain instance from config."""
    from clsplusplus import Brain
    return Brain(user=cfg["user"], api_key=cfg["api_key"], url=cfg["url"])


def _resolve_model(model: str) -> str:
    """Resolve short model names to full IDs."""
    aliases = {
        "gpt4": "gpt-4o",
        "gpt": "gpt-4o",
        "openai": "gpt-4o",
        "claude": "claude-sonnet-4-20250514",
        "sonnet": "claude-sonnet-4-20250514",
        "haiku": "claude-haiku-4-5-20251001",
        "opus": "claude-opus-4-20250514",
        "gemini": "gemini-2.0-flash",
    }
    return aliases.get(model.lower(), model)


def _model_provider(model: str) -> str:
    """Determine provider from model name."""
    m = model.lower()
    if any(m.startswith(p) for p in ("gpt", "o1")):
        return "openai"
    if "claude" in m:
        return "anthropic"
    if "gemini" in m:
        return "google"
    return "openai"  # default


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_init(args, cfg):
    """Interactive setup."""
    print("CLS++ Setup")
    print("=" * 40)

    # API key
    key = args.key or cfg["api_key"]
    if not key:
        key = input("CLS++ API key (from clsplusplus.com/profile.html#keys): ").strip()
    if not key:
        print("No API key provided. Get one at https://www.clsplusplus.com/profile.html#keys")
        return

    # User
    user = args.user or cfg["user"]
    if user == os.environ.get("USER", "default"):
        entered = input(f"Username [{user}]: ").strip()
        if entered:
            user = entered

    # LLM keys
    openai_key = args.openai_key or cfg["openai_key"]
    if not openai_key:
        openai_key = input("OpenAI API key (optional, press Enter to skip): ").strip()

    anthropic_key = args.anthropic_key or cfg["anthropic_key"]
    if not anthropic_key:
        anthropic_key = input("Anthropic API key (optional, press Enter to skip): ").strip()

    google_key = args.google_key or cfg["google_key"]
    if not google_key:
        google_key = input("Google API key (optional, press Enter to skip): ").strip()

    new_cfg = {
        "api_key": key,
        "user": user,
        "url": args.url or cfg["url"],
        "openai_key": openai_key,
        "anthropic_key": anthropic_key,
        "google_key": google_key,
    }

    # Test connection
    print("\nTesting connection...")
    try:
        import httpx
        resp = httpx.get(f"{new_cfg['url'].rstrip('/')}/v1/health", timeout=10)
        if resp.status_code == 200:
            print(f"Connected to {new_cfg['url']}.")
        else:
            print(f"Server responded with {resp.status_code}. Check your URL.")
    except Exception as e:
        print(f"Warning: could not reach server ({e}). Config saved anyway.")

    _save_config(new_cfg)
    print("\nReady! Try: cls learn \"I prefer Python\"")


def cmd_chat(args, cfg):
    """Chat with any LLM — memories auto-captured and injected."""
    import httpx

    model = _resolve_model(args.model or "gpt-4o")
    provider = _model_provider(model)
    url = cfg["url"].rstrip("/")

    # Determine LLM API key
    if provider == "openai":
        llm_key = cfg.get("openai_key", "")
    elif provider == "anthropic":
        llm_key = cfg.get("anthropic_key", "")
    else:
        llm_key = cfg.get("google_key", "")

    if not llm_key:
        print(f"No {provider} API key configured. Run: cls init")
        return

    def send_message(message: str) -> str:
        """Send a message through CLS++ proxy and return response."""
        headers = {"Content-Type": "application/json"}

        if provider == "anthropic":
            # Anthropic format
            headers["x-api-key"] = llm_key
            headers["anthropic-version"] = "2023-06-01"
            if cfg["api_key"]:
                headers["Authorization"] = f"Bearer {cfg['api_key']}"
            body = {
                "model": model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": message}],
            }
            endpoint = f"{url}/v1/messages"
        else:
            # OpenAI format (also works for Gemini via compatible API)
            headers["Authorization"] = f"Bearer {llm_key}"
            body = {
                "model": model,
                "messages": [{"role": "user", "content": message}],
            }
            endpoint = f"{url}/v1/chat/completions"

        # Add user namespace
        body["user"] = cfg["user"]

        resp = httpx.post(endpoint, json=body, headers=headers, timeout=90)
        resp.raise_for_status()
        data = resp.json()

        if provider == "anthropic":
            return data.get("content", [{}])[0].get("text", "")
        else:
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    if args.message:
        # One-shot mode
        message = " ".join(args.message)
        try:
            reply = send_message(message)
            print(f"\n{reply}\n")
        except httpx.HTTPStatusError as e:
            print(f"Error: {e.response.status_code} — {e.response.text[:200]}")
        except Exception as e:
            print(f"Error: {e}")
        return

    # Interactive mode
    print(f"CLS++ Chat (model: {model})")
    print("Type /quit to exit, /memories to list, /who for profile\n")

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not message:
            continue
        if message == "/quit":
            print("Goodbye!")
            break
        if message == "/memories":
            brain = _brain(cfg)
            for m in brain.all():
                print(f"  - {m}")
            continue
        if message == "/who":
            brain = _brain(cfg)
            profile = brain.who()
            print(f"\n{profile.get('summary', 'No profile yet.')}\n")
            continue

        try:
            reply = send_message(message)
            print(f"\nAI: {reply}\n")
        except httpx.HTTPStatusError as e:
            print(f"Error: {e.response.status_code}")
        except Exception as e:
            print(f"Error: {e}")


def cmd_learn(args, cfg):
    """Store a memory."""
    brain = _brain(cfg)
    fact = " ".join(args.fact)
    mid = brain.learn(fact)
    print(f"Learned: \"{fact}\"")


def cmd_ask(args, cfg):
    """Query memories."""
    brain = _brain(cfg)
    question = " ".join(args.question)
    results = brain.ask(question, limit=args.limit)
    if not results:
        print("No matching memories found.")
        return
    for r in results:
        print(f"  - {r}")


def cmd_memories(args, cfg):
    """List all stored memories."""
    brain = _brain(cfg)
    mems = brain.all(limit=args.limit)
    if not mems:
        print("No memories stored yet. Try: cls learn \"I prefer Python\"")
        return
    print(f"{len(mems)} memories:\n")
    for i, m in enumerate(mems, 1):
        print(f"  {i}. {m}")


def cmd_who(args, cfg):
    """Show auto-generated user profile."""
    brain = _brain(cfg)
    profile = brain.who()
    print(f"\nProfile: {profile.get('user', cfg['user'])}")
    print(f"Memories: {profile.get('count', 0)}\n")
    if profile.get("summary"):
        print(profile["summary"])
    print()
    for fact in profile.get("facts", []):
        print(f"  - {fact}")


def cmd_status(args, cfg):
    """Show connection status and memory count."""
    import httpx

    url = cfg["url"].rstrip("/")
    print(f"Server:  {url}")
    print(f"User:    {cfg['user']}")
    print(f"API Key: {'configured' if cfg['api_key'] else 'not set'}")
    print(f"Config:  {CONFIG_PATH}")

    # Health check
    try:
        resp = httpx.get(f"{url}/v1/health", timeout=10)
        data = resp.json()
        print(f"Status:  {data.get('status', 'unknown')}")
    except Exception:
        print("Status:  unreachable")

    # Memory count
    if cfg["api_key"]:
        try:
            brain = _brain(cfg)
            print(f"Memories: {brain.count()}")
        except Exception:
            print("Memories: unable to fetch")

    # LLM keys
    print(f"\nOpenAI:    {'configured' if cfg.get('openai_key') else 'not set'}")
    print(f"Anthropic: {'configured' if cfg.get('anthropic_key') else 'not set'}")
    print(f"Google:    {'configured' if cfg.get('google_key') else 'not set'}")


def cmd_forget(args, cfg):
    """Delete a memory."""
    brain = _brain(cfg)
    fact = " ".join(args.fact)
    ok = brain.forget(fact)
    if ok:
        print(f"Forgotten: \"{fact}\"")
    else:
        print(f"Could not find memory: \"{fact}\"")


def cmd_absorb(args, cfg):
    """Bulk-learn from a file."""
    brain = _brain(cfg)
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return
    content = path.read_text()
    count = brain.absorb(content, source=path.name)
    print(f"Absorbed {count} facts from {path.name}")


def cmd_context(args, cfg):
    """Get LLM-ready context string."""
    brain = _brain(cfg)
    topic = " ".join(args.topic) if args.topic else ""
    ctx = brain.context(topic)
    if ctx:
        print(ctx)
    else:
        print("No relevant context found.")


def cmd_models(args, cfg):
    """List all supported models."""
    print("SUPPORTED MODELS")
    print("=" * 60)
    print()
    print("  OPENAI (requires OPENAI_API_KEY)")
    print("  ---------------------------------")
    print("    gpt-4o           GPT-4o (recommended)")
    print("    gpt-4o-mini      GPT-4o Mini (fast, cheap)")
    print("    gpt-4            GPT-4")
    print("    gpt-4-turbo      GPT-4 Turbo")
    print("    gpt-3.5-turbo    GPT-3.5 Turbo (legacy)")
    print("    o1               OpenAI o1")
    print("    o1-mini          OpenAI o1 Mini")
    print()
    print("  ANTHROPIC (requires ANTHROPIC_API_KEY)")
    print("  --------------------------------------")
    print("    claude           Claude Sonnet (default)")
    print("    claude-sonnet    Claude Sonnet 4")
    print("    claude-haiku     Claude Haiku 4.5 (fast)")
    print("    claude-opus      Claude Opus 4")
    print()
    print("  GOOGLE (requires GOOGLE_API_KEY)")
    print("  --------------------------------")
    print("    gemini           Gemini 2.0 Flash (default)")
    print("    gemini-pro       Gemini Pro")
    print()
    print("  SHORTCUTS")
    print("  ---------")
    print("    gpt    -> gpt-4o")
    print("    sonnet -> claude-sonnet-4-20250514")
    print("    haiku  -> claude-haiku-4-5-20251001")
    print("    opus   -> claude-opus-4-20250514")
    print()
    print("  EXAMPLES")
    print("  --------")
    print('    cls chat -m gpt-4o "What is memory?"')
    print('    cls chat -m claude "What did I just ask GPT?"')
    print('    cls chat -m gemini "Summarize what you know about me"')


# ── Main ──────────────────────────────────────────────────────────────────────

MAIN_HELP = """\
NAME
    cls - CLS++ persistent memory for every AI model

SYNOPSIS
    cls <command> [options] [arguments]

DESCRIPTION
    CLS++ gives every AI model persistent memory. Tell something to GPT-4,
    switch to Claude, it already knows. Memories persist across sessions,
    models, and devices.

COMMANDS
    init              Setup API keys and configuration
    chat              Chat with any LLM (memories auto-captured and injected)
    learn             Store a memory
    ask               Query memories semantically
    memories          List all stored memories
    models            List all supported models and shortcuts
    who               Show auto-generated user profile
    status            Show connection status, memory count, configured keys
    forget            Delete a memory
    absorb            Bulk-learn from a file
    context           Get LLM-ready context string for prompts

QUICK START
    cls init --key YOUR_API_KEY
    cls learn "I prefer Python and dark mode"
    cls ask "What are my preferences?"

CROSS-MODEL MEMORY TRANSFER
    cls chat -m gpt-4o "My name is Raj and I love Python"
    cls chat -m claude "What's my name and what do I love?"
    # -> Claude knows: "Your name is Raj and you love Python."

CONFIGURATION
    Config file: ~/.clsplusplus
    Environment variables: CLS_API_KEY, CLS_BASE_URL, CLS_USER,
                           OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY

SEE ALSO
    cls <command> --help    Detailed help for a specific command
    cls models              List all supported models
    https://www.clsplusplus.com/integrate.html
"""


def main():
    parser = argparse.ArgumentParser(
        prog="cls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=MAIN_HELP,
    )
    parser.add_argument("--user", help="User identifier (overrides config)")
    parser.add_argument("--key", help="CLS++ API key (overrides config)")
    parser.add_argument("--url", help="Server URL (overrides config)")

    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Setup API keys and configuration",
        description="""\
NAME
    cls init - Configure CLS++ API keys and connection

DESCRIPTION
    Interactive setup wizard. Saves configuration to ~/.clsplusplus.
    After init, all other commands work without passing keys.

EXAMPLES
    cls init
    cls init --key cls_live_xxx
    cls init --key cls_live_xxx --openai-key sk-xxx --user raj
""")
    p_init.add_argument("--key", dest="key", help="CLS++ API key")
    p_init.add_argument("--user", help="Username")
    p_init.add_argument("--url", help="Server URL")
    p_init.add_argument("--openai-key", dest="openai_key", help="OpenAI API key")
    p_init.add_argument("--anthropic-key", dest="anthropic_key", help="Anthropic API key")
    p_init.add_argument("--google-key", dest="google_key", help="Google API key")

    # chat
    p_chat = sub.add_parser("chat", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Chat with any LLM — memories transfer across models",
        description="""\
NAME
    cls chat - Chat with any LLM with persistent cross-model memory

DESCRIPTION
    Send a message to any supported model. CLS++ automatically:
      1. Retrieves relevant memories and injects them into the prompt
      2. Forwards the request to the LLM (OpenAI, Anthropic, Google)
      3. Captures the conversation as new memories
      4. Returns the LLM's response

    Memories persist across models. Tell GPT-4 your name, then ask
    Claude — it already knows.

    Omit the message to enter interactive mode.

    Run 'cls models' to see all supported models.

EXAMPLES
    cls chat -m gpt-4o "My name is Raj and I love Python"
    cls chat -m claude "What's my name?"
    cls chat -m gemini "Summarize what you know about me"
    cls chat                          # interactive mode
    cls chat -m haiku "Quick question" # use shortcut names
""")
    p_chat.add_argument("--model", "-m", default="gpt-4o",
        help="Model name or shortcut (default: gpt-4o). Run 'cls models' for full list")
    p_chat.add_argument("message", nargs="*", help="Message (omit for interactive mode)")

    # learn
    p_learn = sub.add_parser("learn", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Store a memory",
        description="""\
NAME
    cls learn - Teach CLS++ a fact to remember

DESCRIPTION
    Store any fact as a persistent memory. The memory is available
    to all models and persists across sessions.

EXAMPLES
    cls learn "I prefer Python and dark mode"
    cls learn "My project is called Atlas and uses FastAPI"
    cls learn "Meeting with Alice on Friday at 3pm"
""")
    p_learn.add_argument("fact", nargs="+", help="Fact to remember")

    # ask
    p_ask = sub.add_parser("ask", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Query memories semantically",
        description="""\
NAME
    cls ask - Query memories using natural language

DESCRIPTION
    Semantic search across all stored memories. Returns the most
    relevant matches ranked by confidence.

EXAMPLES
    cls ask "What are my coding preferences?"
    cls ask "What meetings do I have?" -n 10
    cls ask "Tell me about my project"
""")
    p_ask.add_argument("question", nargs="+", help="Question to ask")
    p_ask.add_argument("--limit", "-n", type=int, default=5, help="Max results (default: 5)")

    # memories
    p_mems = sub.add_parser("memories", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="List all stored memories",
        description="""\
NAME
    cls memories - List all stored memories

EXAMPLES
    cls memories
    cls memories -n 100
""")
    p_mems.add_argument("--limit", "-n", type=int, default=50, help="Max results (default: 50)")

    # models
    sub.add_parser("models", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="List all supported models and shortcuts",
        description="""\
NAME
    cls models - Show all supported LLM models

DESCRIPTION
    Lists every model CLS++ can route to, organized by provider,
    with shortcut names for convenience.
""")

    # who
    sub.add_parser("who", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Show auto-generated user profile",
        description="""\
NAME
    cls who - Show auto-generated user profile

DESCRIPTION
    Generates a structured profile from all stored memories.
    Shows facts, summary, and memory count.

EXAMPLES
    cls who
""")

    # status
    sub.add_parser("status", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Show connection status, memory count, configured keys",
        description="""\
NAME
    cls status - Show connection and configuration status

DESCRIPTION
    Displays server URL, user, API key status, health check result,
    memory count, and which LLM provider keys are configured.

EXAMPLES
    cls status
""")

    # forget
    p_forget = sub.add_parser("forget", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Delete a memory",
        description="""\
NAME
    cls forget - Delete a memory

DESCRIPTION
    Remove a specific memory by its text or ID. Supports GDPR
    right-to-be-forgotten compliance.

EXAMPLES
    cls forget "I work at Google"
    cls forget "old-memory-id-here"
""")
    p_forget.add_argument("fact", nargs="+", help="Fact text or memory ID to forget")

    # absorb
    p_absorb = sub.add_parser("absorb", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Bulk-learn from a file",
        description="""\
NAME
    cls absorb - Bulk-learn facts from a text file

DESCRIPTION
    Reads a file, splits into sentences/paragraphs, and stores
    each as a separate memory. Good for importing notes, documents,
    or conversation logs.

EXAMPLES
    cls absorb meeting-notes.txt
    cls absorb ~/Documents/project-brief.md
""")
    p_absorb.add_argument("file", help="Path to text file")

    # context
    p_context = sub.add_parser("context", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Get LLM-ready context string for prompts",
        description="""\
NAME
    cls context - Generate context string for LLM prompts

DESCRIPTION
    Returns a formatted block of relevant memories suitable for
    injecting into a system prompt. Use this when building custom
    LLM integrations.

EXAMPLES
    cls context "coding help"
    cls context "meeting preparation"
""")
    p_context.add_argument("topic", nargs="*", help="Topic to get context for")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    # Load config, apply CLI overrides
    cfg = _load_config()
    if getattr(args, "key", None) and args.command != "init":
        cfg["api_key"] = args.key
    if getattr(args, "user", None) and args.command != "init":
        cfg["user"] = args.user
    if getattr(args, "url", None) and args.command != "init":
        cfg["url"] = args.url

    commands = {
        "init": cmd_init,
        "chat": cmd_chat,
        "learn": cmd_learn,
        "ask": cmd_ask,
        "memories": cmd_memories,
        "models": cmd_models,
        "who": cmd_who,
        "status": cmd_status,
        "forget": cmd_forget,
        "absorb": cmd_absorb,
        "context": cmd_context,
    }

    fn = commands.get(args.command)
    if fn:
        fn(args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
