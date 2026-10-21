# Question set

## `in_scope.json` (you supply this)

Real traveler questions that the approved documents in `data/` **can** answer.
`scripts/calibrate_evidence.py` uses them, together with the must-refuse traps in
`data/unknown_traps.json`, to choose the strong/medium evidence thresholds.

Aim for 25 or more questions that cover every document (gates, terminals,
amenities, hours, transport, security, FAQ), phrased the way people actually
ask (short, spoken, with a few mis-hearings). The calibration script refuses to
write thresholds with fewer than 10.

```json
{
  "questions": [
    {
      "id": "gate-c22-where",
      "question": "Where is gate C22?",
      "expected_source": "gates.md",
      "must_contain": ["Terminal 2", "Concourse C"]
    }
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | no | Short label shown in reports |
| `question` | yes | The question text |
| `expected_source` | no | Substring of the source document that should be retrieved (e.g. `gates.md`); the report flags retrieval misses |
| `must_contain` | no | Facts the grounded answer must include (used by the Stage 4 evaluation harness) |

`in_scope.example.json` shows the format. It is **not** read by any script.

## Calibrating

```powershell
python scripts/calibrate_evidence.py            # report only
python scripts/calibrate_evidence.py --write    # saves config/evidence_thresholds.json (calibrated: true)
```

Until `calibrated` is true the server logs a warning at startup and uses
conservative placeholder thresholds (it prefers a safe fallback over a risky answer).
