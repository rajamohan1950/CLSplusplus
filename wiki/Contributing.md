# Contributing to CLS++

See the main [CONTRIBUTING.md](../.github/CONTRIBUTING.md) in the repository.

---

## Quick Summary

1. **Fork** the repo
2. **Create a branch**: `git checkout -b feature/your-feature`
3. **Make changes** — follow existing style
4. **Add tests**: `pytest tests/`
5. **Open a PR**

---

## Development Setup

```bash
git clone https://github.com/rajamohan1950/CLSplusplus.git
cd CLSplusplus
pip install -e ".[dev]"
docker compose up -d redis postgres minio
pytest tests/
```

---

## Project Structure

- `src/clsplusplus/` — Core API, stores (L0–L3), plasticity, sleep cycle
- `tests/` — Pytest tests
- `docs/` — Documentation
- `infrastructure/` — AWS, Azure templates

---

[← SaaS & Pricing](SaaS-and-Pricing) | [Home →](Home)
