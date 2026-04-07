# CLS++ Memory Plugin for Claude Code

Persistent cross-session memory for Claude Code. Every session remembers what happened in previous sessions.

## What It Does

- **Session Start**: Automatically recalls relevant memories from your previous Claude Code sessions
- **After Each Response**: Stores what you worked on (user question, files modified, tools used) as a memory
- **Next Session**: Those memories come back, giving Claude full context of your development history

No configuration beyond an API key. Zero friction. Memories are stored in the `claude-code` namespace on your CLS++ account.

## Setup

### 1. Get a CLS++ Account

Sign up at [clsplusplus.com](https://www.clsplusplus.com) and get your API key from the profile page.

### 2. Install the Plugin

```bash
# Add the marketplace
/plugin marketplace add rajamohan1950/CLSplusplus

# Install
/plugin install cls-memory
```

### 3. Set Your API Key

Add to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
export CLS_API_KEY=cls_live_xxxxx
```

Restart your terminal. Done.

## Usage

The plugin works automatically. No commands needed.

**Check status**: Use `/cls-memory` to verify your connection and see stored memories.

**Manual recall**: `/cls-memory recall` to manually fetch memories mid-session.

**Manual store**: `/cls-memory store <text>` to manually save something important.

## How It Works

| Event | What Happens |
|-------|-------------|
| Session starts | Fetches last 15 relevant memories from CLS++ API |
| Claude responds | Extracts summary from transcript, stores as memory |
| Next session | Recalled memories are injected as context |

Memories are stored with:
- `namespace`: `claude-code` (isolated from your app data)
- `source`: `claude-code-plugin`
- `salience`: 0.7 (moderate importance, retrievable but not overwhelming)

## Security

- API key is read from environment variable only, never stored in files
- All communication uses HTTPS
- Hooks fail silently (exit 0) if API is unreachable, so Claude Code is never blocked
- 10-second timeout on all API calls

## Requirements

- Claude Code 1.0.0 or later
- Python 3.9+ (for hook scripts)
- CLS++ account with API key ([sign up](https://www.clsplusplus.com))

## License

Apache 2.0 - same as CLS++
