# Floor plan evaluation fixtures

Place one SVG per case under a named subfolder:

```
tests/floorplans/
  simple_apartment/model.svg
  multi_room_apartment/model.svg
  complex_architectural/model.svg
```

Supported input names: `model.svg` (preferred), or any `*.svg` / `*.png` / `*.jpg`.

Run evaluation with:

```bash
python scripts/evaluate_pipeline.py --input-dir tests/floorplans --out-dir tests/results
```
