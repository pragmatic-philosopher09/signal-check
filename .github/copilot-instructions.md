# Copilot instructions — Signal Check

`COPILOT_BRIEF.md` in the repo root is the specification for this project. Sections 0–9 are binding for all code you write; section 10 contains the phased prompts the developer will give you one at a time.

Key rules (full detail in the brief):

- Deterministic statistics decide every verdict; the LLM only narrates validated JSON. No ML model for the verdict, no forecasting.
- Every threshold lives in `config.yaml`, never as a literal in engine code.
- Python 3.12, type hints everywhere, small pure functions, docstrings that state the statistical meaning of each check. Code must pass `ruff`, `mypy` and `pytest`.
- Seeded randomness only. Tests never hit the network (mock HTTP).
- Never interpolate silently; record gaps, imputation, truncation and dropped partial periods in `meta.caveats`.
- Adapter failures degrade to a per-card message; the app never crashes because one source failed.
- Cache every external call (6h TTL). Store only aggregates; never persist author names or post text.
- No secrets in code; use `.env` / `st.secrets` via `get_secret()`.
- Official APIs only; no X scraping; identify the app in the User-Agent.
- Only build what the current phase asks for.
