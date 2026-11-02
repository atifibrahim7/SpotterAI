# Freight Rate Prediction Challenge

See `freight-rate-ml-assessment-1-.pdf` for the assessment instructions, and `report.md` for my approach and results.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

Walkthrough with charts: `freight_rate_model.ipynb` (select the "SpotterAI (.venv)" kernel). It imports the pipeline from `train.py`.

`train.py` writes `validation_predictions.csv`, fills `december-chart-inputs.csv`, and saves metrics to `metrics.json`. Data files are in the repo root (names use hyphens, not the `data/` paths below).

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link