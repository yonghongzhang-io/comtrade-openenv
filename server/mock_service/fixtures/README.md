# Mock Fixtures

This directory is intentionally empty — no JSONL fixture files are needed.

The mock service generates all trade records **deterministically at runtime**
using a seeded PRNG keyed on `(task_id, reporter, partner, flow, hs, year)`:

```python
seed = sha256(f"{task_id}:{reporter}:{partner}:{flow}:{hs}:{year}")
rng  = random.Random(seed)
rows = [{"year": ..., "tradeValue": rng.randint(...), ...} for i in range(total_rows)]
```

This means:
- The same query always returns the same data (reproducible across runs)
- No large fixture files need to be stored in git
- Any test or evaluation can be replicated without external data

## Schema

Each generated row contains:

| Field       | Type   | Description                    |
|-------------|--------|--------------------------------|
| `year`      | int    | Trade year                     |
| `reporter`  | str    | Reporter country code (ISO M49)|
| `partner`   | str    | Partner country code (ISO M49) |
| `flow`      | str    | Trade flow (`M`=import, `X`=export) |
| `hs`        | str    | HS product code                |
| `cmdCode`   | str    | Commodity code (same as hs)    |
| `tradeValue`| int    | Trade value (USD)              |
| `netWeight` | int    | Net weight (kg)                |
| `qty`       | int    | Quantity                       |
| `record_id` | str    | Unique record identifier       |

## Totals Rows (T7 only)

In `totals_trap` mode the mock service prepends one synthetic totals row to
every page response. Totals rows can be identified by **either** field name
(both are set for compatibility):

- `isTotal = true`  (camelCase)
- `is_total = true` (snake_case)
- `partner = "WLD"`
- `hs = "TOTAL"`

Agents must filter **all** rows matching these markers before submitting.

## To Add Custom Fixtures

If you want the mock to serve hand-crafted data instead of generated rows,
place a JSONL file named `<task_id>.jsonl` in this directory. The loader
checks for fixture files first and falls back to procedural generation:

```python
def _get_base_rows(task_id, q, total_rows):
    fixture = _load_fixture(task_id)   # returns None if no file
    if fixture:
        return fixture
    return _generate_rows(task_id, q, total_rows)
```
