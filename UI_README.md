# HR Attrition UI

This Streamlit app wraps the full notebook workflow into an interactive dashboard:

- Data overview
- Single employee risk prediction
- SHAP-based explainability
- Fairness snapshot (Sex, RaceDesc)
- Frugal AI monitor (reads `emissions.csv`)

## Run

```powershell
pip install -r requirements_ui.txt
streamlit run app.py
```

Open the URL shown in terminal (usually http://localhost:8501).

## Notes

- The app expects `HRDataset_v14.csv` in the same folder.
- If `emissions.csv` exists, the Frugal AI tab shows the latest records.
- The model is retrained in memory from the dataset (cached by Streamlit).
