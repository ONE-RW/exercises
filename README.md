# Student Status Predictor

Streamlit app that predicts Pass/Fail for Rwandan O-level students from subject marks.

## Files
- `app.py` - the web app
- `requirements.txt` - dependencies (pin `scikit-learn` to the version your notebook printed)
- `student_status_pipeline.joblib` - **you create this** by running the notebook's last cell, then download it from Kaggle (Output panel)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## Deploy option A: Streamlit Community Cloud (free, easiest)
1. Create a GitHub repo and upload `app.py`, `requirements.txt`, `student_status_pipeline.joblib`.
   (GitHub's file limit is 100 MB. If the model is bigger, see "Model size" below.)
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Choose the repo, branch `main`, main file `app.py`, then **Deploy**.
4. You get a public link like `https://your-app.streamlit.app` to share.

## Deploy option B: Hugging Face Spaces (free, handles larger files)
1. Create a new Space at https://huggingface.co/new-space, SDK = **Streamlit**.
2. Upload the same three files (large files are handled with Git LFS automatically in the web uploader).
3. The Space builds and serves the app at `https://huggingface.co/spaces/<you>/<space>`.

## Model size
If the Random Forest wins, the saved file can be large. Options: use the Logistic Regression
(its CV score is almost identical and the file is tiny), or train the forest with
`RandomForestClassifier(n_estimators=100, min_samples_leaf=5, ...)`.

## Notes
- Blank marks are treated as missing and imputed with the training medians, as in the notebook.
- The app accepts marks from 0 to 100 only.
