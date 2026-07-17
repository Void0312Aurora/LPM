# Scripts

Scripts are grouped by pipeline role:

- `data/`: deterministic data construction and conversion.
- `llm/`: external-LLM cleaning/enrichment requests and merge logic.
- `annotation/`: local human annotation server, export logic, and Web GUI.
- `mvp/`: MVP unit preparation, embedding extraction, split, training, and evaluation.
- `demo/`: demo-passage extraction utilities.

Run scripts from the repository root, for example:

```bash
python scripts/llm/llm_fill_clean_events.py
```

Start the human annotation GUI:

```bash
python scripts/annotation/annotate_clean_events.py serve --annotator "$USER"
```
