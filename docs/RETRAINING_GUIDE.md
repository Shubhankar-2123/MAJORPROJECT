# Feedback → Retraining Workflow (Offline)

**Goal**: Use high-quality user feedback to improve models safely without overwriting existing versions.

## 1) Collect feedback (already in app)
- Feedback is stored in SQLite: `feedback` table.
- Each entry links `prediction_id`, `original_text`, `correction_text`, `processed`.

## 2) Export retraining candidates
**Criteria (recommended):**
- Confidence ≥ 0.80
- Same correction repeated at least 3 times
- User-corrected labels differ from predictions

**Export query (example):**
```sql
SELECT original_text, correction_text, COUNT(*) AS cnt
FROM feedback
WHERE processed = 0
GROUP BY original_text, correction_text
HAVING cnt >= 3;
```

## 3) Prepare dataset
- Merge feedback corrections with the original dataset.
- Create a `retraining.csv` with fields: `input_path`, `label`, `source`.
- Keep old training data intact; append new samples.

## 4) Retrain offline
- Run training scripts in `scripts/`.
- Save new model versions with a suffix:
  - `static_model_v1_1.pth`
  - `dynamic_model_v1_1.pth`
- Do **not** overwrite working models.

## 5) Validate
- Evaluate on a fixed validation set.
- Compare accuracy and confusion metrics vs previous model.

## 6) Deploy new version
- Update the loader to point to the new model files.
- Keep old versions available for rollback.

## 7) Mark feedback as processed
- After deployment, mark feedback rows used in retraining:
```sql
UPDATE feedback SET processed = 1 WHERE id IN (...);
```

## Notes
- Never auto-retrain live models.
- Always keep an audit trail of dataset version + model version.
