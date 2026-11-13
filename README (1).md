# Clothing Shop Sales and Customer Feedback Assistant

A five-tab Streamlit app for a small clothing shop owner: demand estimate, product profile groups, sample image recognition, customer comment analysis and a weekly review. **All data is generated classroom data. Results are demonstrations, not a real sales forecast.**

## Run steps
1. Train once in free Google Colab with `Project02_Full_Notebook.ipynb`. Download `project02_outputs.zip` (the `models/` folder) and `project02_reports.zip`.
2. Put the repository together (see layout below). Copy `reports/requirements_pinned.txt` versions into `requirements.txt` so the saved models load.
3. Local run: `pip install -r requirements.txt` then `streamlit run app.py`.
4. Deploy: upload to a free GitHub repository, then create the app on free Streamlit Community Cloud (branch + `app.py`, Python 3.11 or 3.12).
5. Optional AI summary (free Groq plan only): in the Secrets panel add
   `GROQ_API_KEY = "..."` and `GROQ_MODEL = "openai/gpt-oss-20b"`. The app works fully without it (Offline summary is the default).

## Repository layout
```
app.py
requirements.txt
README.md
.gitignore
.streamlit/config.toml        (pastel theme)
models/   supervised.joblib, clusters.joblib, clusters.json, images.keras, metadata.json, metrics.json
data/     main.csv, feedback.csv, example_inputs.csv, meta.json, topic_keywords.json, prompt_test_cases.json
nltk_data/sentiment/vader_lexicon.zip   (from the supplied pack)
source_notices/                         (from the supplied pack - keep the attribution)
sample_images/                          (from the supplied pack)
```
Do not upload `images.npz`, ZIP files or `.streamlit/secrets.toml`.

## Limits
- Demand numbers come from a generated teaching recipe; they do not show the app can forecast a real shop or recommend discounts.
- KMeans groups are product scenarios, not customer segments; group numbers are arbitrary.
- The image model is trained on small grey Fashion MNIST items and may fail on coloured phone photos or several garments. Always check the suggestion.
- Sentiment (VADER) and keyword topics are English only and miss sarcasm, negation and mixed opinions.
- Free hosting, Colab and the Groq free plan have usage limits. If a limit blocks the live demo, run the app locally and record the screen.

Attribution: images from Fashion MNIST; comments are authored classroom examples. Keep the supplied `source_notices/`.
