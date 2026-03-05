# GitHub Repository Setup Guide

Make your CLS++ repo look **beautiful and cutting-edge**. Follow these steps to configure every section.

---

## 1. About Section (Right sidebar)

**Settings:** Repo → **About** (gear icon) → Edit

| Field | Suggested Value |
|-------|-----------------|
| **Description** | Brain-inspired, model-agnostic persistent memory for LLMs. Switch AI models. Never lose context. |
| **Website** | https://clsplusplus.onrender.com |
| **Topics** | `llm`, `memory`, `ai`, `cls`, `continuous-learning`, `embeddings`, `fastapi`, `python`, `neuroscience`, `rag` |

---

## 2. Social Preview Image

**Settings:** Repo → **General** → Social preview → Upload

- Use `.github/social-preview.png` (1280×640 px recommended)
- Or create at [Canva](https://canva.com) / [Figma](https://figma.com): "CLS++ — Memory for LLMs" on dark background (#0a0a0f) with indigo accent (#6366f1)

---

## 3. Projects

**Create a project:** Repo → **Projects** → New project → **Board** (or Table)

**Suggested columns:**
- **Backlog** — New ideas, enhancements
- **To Do** — Ready to work on
- **In Progress** — Active work
- **Done** — Completed

**Suggested issues to add:**
- LangChain integration
- LangGraph integration
- Vercel AI SDK adapter
- Stripe billing hook
- HIPAA compliance review

---

## 4. Wiki

Wiki content is in the `wiki/` folder. To publish:

```bash
# Clone the wiki repo (create first page via GitHub UI if empty)
git clone https://github.com/rajamohan1950/CLSplusplus.wiki.git
cd CLSplusplus.wiki

# Copy content from main repo
cp ../wiki/*.md .

# Commit and push
git add .
git commit -m "Add wiki pages"
git push origin main
```

Or use the script: `./scripts/push-wiki.sh`

---

## 5. Discussions

**Settings:** Repo → **General** → Features → ✅ Discussions

Enable for:
- Q&A
- Ideas
- General

---

## 6. Branch Protection (Optional)

**Settings:** Repo → **Branches** → Add rule

- Branch name: `main`
- ✅ Require pull request before merging
- ✅ Require status checks (CI)
- ✅ Require branches to be up to date

---

## 7. Secrets (for CI)

**Settings:** Repo → **Secrets and variables** → Actions

Add if needed for deployment:
- `RENDER_DEPLOY_HOOK` (for auto-deploy on push)
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (for demo, if not in Render env)

---

## 8. GitHub Pages (Optional)

If you want docs at `https://rajamohan1950.github.io/CLSplusplus`:

**Settings:** Repo → **Pages** → Source: Deploy from branch → `main` → `/website` or `/docs`

---

## Quick Checklist

- [ ] About: description, website, topics
- [ ] Social preview image
- [ ] Projects: create board, add initial issues
- [ ] Wiki: push content from `wiki/` folder
- [ ] Discussions: enable
- [ ] Branch protection (optional)
- [ ] CI passing (workflows in `.github/workflows/`)
