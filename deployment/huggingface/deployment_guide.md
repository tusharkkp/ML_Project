# 100% Free Hugging Face Deployment Guide (Gradio SDK)

## AI-Based Food Inflation Prediction and Household Affordability Monitoring System

> **Key Takeaway:**
>
> - Hugging Face recently made the Docker SDK require a paid PRO plan ($9/mo).
> - **Gradio is Hugging Face's official 100% FREE Python SDK.**
> - By selecting **Gradio**, you get **CPU Basic (2 vCPU · 16 GB RAM)** for **$0.00 / month (no payment, no credit card required)**.
> - We have built a native Gradio application ([`gradio_app.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/gradio_app.py) / [`deployment/huggingface/app.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/deployment/huggingface/app.py)) containing all 4 modules (Price Forecaster, Affordability, Benchmarks, and Viva Guide).

---

### Step 1: Create a Free Hugging Face Space

1. Log in to [huggingface.co](https://huggingface.co/) and click **New Space** (or visit `https://huggingface.co/new-space`).
2. Configure your Space:
   - **Space Name:** `food-inflation-monitor` (or your preferred name)
   - **License:** `mit`
   - **Select the Space SDK:** Choose **Gradio** _(100% Free)_
   - **Space Hardware:** Keep default **CPU Basic • 2 vCPU • 16 GB • Free** ($0.00/month)
   - **Space Visibility:** **Public**
3. Click **Create Space**.

---

### Step 2: Upload Files to the Space

Upload the following files into your Hugging Face Space repository:

1. **`app.py`**: Copy from [`deployment/huggingface/app.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/deployment/huggingface/app.py) (this is the Gradio app).
2. **`requirements.txt`**: The project's [`requirements.txt`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/requirements.txt) (contains `gradio`, `scikit-learn`, `pandas`, `joblib`, etc.).
3. **`README.md`**: Copy from [`deployment/huggingface/README.md`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/deployment/huggingface/README.md) (contains `sdk: gradio`).
4. **`models/`** directory: Upload your trained models (`unified_food_price_model.joblib`, `unified_model_metadata.json`, and crop models).
5. **`src/`** directory: Upload `inference.py` and `affordability.py`.
6. **`reports/figures/`** directory: Upload the diagnostic plots.

---

### Step 3: Uploading via Web Browser UI (Easiest)

1. Open your newly created Space on Hugging Face.
2. Click on the **Files** tab.
3. Click **Add file** $\to$ **Upload files**.
4. Drag and drop the files listed above.
5. Click **Commit changes to main**.

---

### Step 4: Live Build & Launch

- Hugging Face automatically detects `app.py` and installs dependencies from `requirements.txt`.
- Within 60 seconds, the Space status transitions to **Running**.
- Your web application is now **LIVE** with a permanent public link:
  `https://huggingface.co/spaces/<your-username>/food-inflation-monitor`

---

## Alternative: Free Streamlit Deployment on Streamlit Community Cloud

If you specifically want the **Streamlit** user interface:

- **Streamlit Community Cloud** ([share.streamlit.io](https://share.streamlit.io/)) is **100% free forever** for public GitHub repositories.
- Simply connect your GitHub repository, select `app.py`, and click Deploy. It launches your Streamlit dashboard for free with zero configuration.
