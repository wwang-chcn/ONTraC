# API Reference Generation

This note is for maintainers who regenerate API docs from ONTraC source code.

## Output

- Website-ready API pages are generated into `docs/api_reference/`.
- The landing page is `docs/api_reference/index.md`.

## Regenerate

```bash
python scripts/generate_api_reference.py
```

Optional flags:

```bash
python scripts/generate_api_reference.py --hide-private
python scripts/generate_api_reference.py --out-dir docs/api_reference
```
