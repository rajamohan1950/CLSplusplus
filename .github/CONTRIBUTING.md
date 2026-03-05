# Contributing to CLS++

Thank you for your interest in contributing to CLS++. We welcome contributions from the community.

## Code of Conduct

By participating, you agree to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

- Use the [Bug Report](https://github.com/rajamohan1950/CLSplusplus/issues/new?template=bug_report.md) template
- Include steps to reproduce, environment details, and logs

### Suggesting Features

- Use the [Feature Request](https://github.com/rajamohan1950/CLSplusplus/issues/new?template=feature_request.md) template
- If your idea maps to a neuroscientific concept (consolidation, forgetting, etc.), we'd love to hear about it

### Pull Requests

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feature/your-feature`
3. **Make changes** — follow existing code style
4. **Add tests** for new functionality
5. **Run tests**: `pytest tests/`
6. **Commit** with clear messages: `feat: add X`, `fix: resolve Y`
7. **Push** and open a PR

### Development Setup

```bash
git clone https://github.com/rajamohan1950/CLSplusplus.git
cd CLSplusplus
pip install -e ".[dev]"
docker compose up -d redis postgres minio
pytest tests/
```

### Project Structure

- `src/clsplusplus/` — Core API, stores (L0–L3), plasticity, sleep cycle
- `tests/` — Pytest tests
- `docs/` — Documentation
- `infrastructure/` — AWS, Azure deployment templates

### Style

- Python: Black-compatible, type hints where helpful
- Docstrings: Google style
- Commits: [Conventional Commits](https://www.conventionalcommits.org/) preferred

## Questions?

Open a [Discussion](https://github.com/rajamohan1950/CLSplusplus/discussions) or an issue.
