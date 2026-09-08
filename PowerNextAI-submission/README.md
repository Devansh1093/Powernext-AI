# PowerNext-AI Screening Submission

This project reads `Dataset.xlsx`, identifies the labeled and unlabeled sheets, validates the data, engineers numeric operating and sensor features, evaluates models with cross-validation, trains the selected models, generates predictions, writes a methodology note and summary, validates the submission schema, and packages the reproducible submission.

## Run

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py
```

The configured project environment can also be used directly:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

Outputs are written to `outputs/`, reports to `reports/`, and the final archive to `PowerNextAI-submission.zip`. The generated submission uses the workbook's sample schema: `Test_ID`, `Predicted_Reference_Parameter`, and `Validity_Label`.

## Method

Reference parameters are predicted with a cross-validated Extra Trees regressor. Validity is predicted with a balanced Extra Trees classifier; its decision threshold is selected from out-of-fold probabilities. Median imputation and missingness indicators are fitted inside each model pipeline. Test IDs are never model features, and no test labels are used.

Validation uses five-fold shuffled K-fold regression and five-fold stratified classification. Metrics in `outputs/summary.json` and `reports/model_comparison.csv` are generated from the actual run, not manually entered. The workbook contains 1,000 labeled training records and 350 unlabeled test records; no duplicate IDs or exact duplicate rows were found.

## Assumptions and limitations

Rows are treated as independent because no grouping key beyond `Test_ID` is provided. The model assumes the hidden data preserves the workbook's column meanings and broadly similar operating range. Validation metrics are estimates and do not guarantee hidden-test performance.
