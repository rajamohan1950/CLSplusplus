# CLS++ Product Video Script — 90 Seconds

## TARGET: Developer-focused, show don't tell, fast-paced

---

### SCENE 1: THE PROBLEM (0:00 - 0:15)

**[Screen recording: Split screen — ChatGPT on left, Claude on right]**

NARRATOR: "You tell ChatGPT you're a Python developer. Switch to Claude — it has no idea."

**[Type in ChatGPT: "I'm a Python developer at Google"]**
**[ChatGPT responds warmly]**
**[Switch to Claude tab: "What programming language do I use?"]**
**[Claude: "I don't have any information about you"]**

TEXT OVERLAY: "Every AI forgets you."

---

### SCENE 2: THE FIX — ONE LINE (0:15 - 0:35)

**[VS Code opens. Python file visible.]**

NARRATOR: "Fix it in one line."

**[Show the code change — red strikethrough → green highlight]**

```python
# Before:
client = OpenAI()

# After:
client = OpenAI(base_url="https://your-cls-server/v1")
```

TEXT OVERLAY: "One line. Zero SDK. Instant memory."

NARRATOR: "That's it. Your app now remembers every user — forever."

**[Show terminal: user sends message → CLS++ injects memory → GPT-4 responds with personalization]**

---

### SCENE 3: THE MAGIC — CROSS-MODEL (0:35 - 0:55)

**[Split screen again: GPT-4 left, Claude right]**

NARRATOR: "Tell GPT-4 you're vegetarian and live in Hyderabad."

**[GPT-4 conversation: "I'm vegetarian, I live in Hyderabad"]**

NARRATOR: "Now switch to Claude."

**[Claude: "What do you know about me?"]**
**[Claude responds: "Hi Raj! You're vegetarian, live in Hyderabad..."]**

TEXT OVERLAY: "Memory crosses models. Zero data loss."

NARRATOR: "Claude knows everything GPT-4 learned. No code changes."

---

### SCENE 4: THE BRAIN SDK (0:55 - 1:10)

**[VS Code with Python code, typing in real-time]**

NARRATOR: "Want full control? Four methods."

```python
from clsplusplus import Brain

brain = Brain("alice")
brain.learn("Prefers dark mode")
brain.ask("What theme?")   # → ["Prefers dark mode"]
brain.context("UI help")   # → LLM-ready prompt
brain.forget("old fact")   # → GDPR delete
```

TEXT OVERLAY: "4 methods. 0 config. ∞ models."

NARRATOR: "Learn. Ask. Context. Forget. That's the entire API."

---

### SCENE 5: THE PLATFORM (1:10 - 1:30)

**[Quick montage — each screen for 3 seconds]**

1. **Admin Dashboard** — KPIs: 1,247 users, $5,839 revenue, 100% margin
2. **User Buckets** — Free / Pro / Business / Enterprise cards
3. **Metrics** — 40+ metering points, live charts
4. **Extension Analytics** — DAU 25, installs, site usage
5. **Permissions** — RBAC roles, groups, page-level access
6. **User Dashboard** — tier, usage, API keys, upgrade CTA

NARRATOR: "Full SaaS platform. User management. Billing. RBAC. 40+ metrics. Built in."

---

### SCENE 6: THE CLOSE (1:30 - 1:45)

**[Dark screen with CLS++ logo]**

TEXT OVERLAY (large):
```
CLS++
Switch AI Models. Never Lose Context.
```

TEXT OVERLAY (smaller):
```
1 line to integrate
4 methods in the SDK
40+ metrics in the admin
Free tier: 1,000 ops/month

clsplusplus.com/getting-started
```

NARRATOR: "CLS++. The memory layer for every AI model."

**[GitHub stars counter animating up]**

---

## PRODUCTION NOTES

- **Music**: Upbeat electronic, subtle (like Linear or Vercel product videos)
- **Pace**: Fast cuts, no pause longer than 3 seconds
- **Font**: DM Sans (matches the product)
- **Colors**: Dark background (#0a0a0f), accent indigo (#6366f1), green for success (#22c55e)
- **Screen recordings**: Use the actual product at localhost:8181
- **Terminal**: Use iTerm2 with dark theme, JetBrains Mono font
- **Voice**: Male, confident, slightly fast (like Fireship.io style)

## TOOLS TO PRODUCE

1. **Screen recordings**: OBS Studio or ScreenFlow
2. **Editing**: DaVinci Resolve (free) or Final Cut Pro
3. **Text overlays**: Built into the editor
4. **Voiceover**: Record yourself or use ElevenLabs AI voice
5. **Music**: Epidemic Sound or Artlist (royalty-free)

## ALTERNATIVE: AI-GENERATED VIDEO

Use one of these to auto-generate from this script:
- **Synthesia** — AI avatar narrates over screen recordings
- **HeyGen** — Same concept, good for product demos
- **Loom** — Record yourself walking through the product
- **Cap** — Free screen recorder with built-in editing
