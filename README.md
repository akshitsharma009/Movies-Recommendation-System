# 🎬 Movie Recommendation System

An end-to-end **content-based movie recommendation system** that suggests movies similar to a
user's favorite choice using natural language processing (TF-IDF) and cosine similarity. The app is
deployed as an interactive web app with Streamlit.

🔗 **Live app:** https://movies-recommendation-system-fgg2ekv383qu3xhmrc5gva.streamlit.app/

## 📌 Overview

The system recommends movies based on similarity in:

- Genres
- Keywords
- Cast
- Director
- Overview

It uses **TF-IDF vectorization** to convert textual metadata into numerical vectors and **cosine
similarity** to measure how close two movies are.

## ⚙️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, scikit-learn, Streamlit
- **ML:** TF-IDF vectorization, cosine similarity
- **Deployment:** Streamlit Community Cloud
- **Version control:** Git & GitHub

## 🧠 How It Works

1. Movie metadata is cleaned and combined into a single text feature.
2. TF-IDF converts the text into numerical feature vectors.
3. Cosine similarity measures similarity between movies.
4. The most similar movies are ranked and returned.
5. Results are displayed in an interactive Streamlit UI.

## 🚀 Features

- Content-based movie recommendation
- Fuzzy matching for movie names (typo-tolerant search)
- Adjustable number of recommendations
- Clean, modern Streamlit UI

## 📂 Project Structure

```
movies-recommendation-system/
│
├── app.py             # Streamlit application
├── movies.csv         # Dataset (TMDB 5000)
├── requirements.txt   # Python dependencies
├── runtime.txt        # Python version for Streamlit Cloud
├── notebook.ipynb     # Exploratory notebook
└── README.md          # Project documentation
```

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/akshitsharma009/movies-recommendation-system.git
cd movies-recommendation-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## 📈 Example

Select **Inception** → the system analyzes metadata similarity and recommends movies such as
*Interstellar*, *The Dark Knight*, and other Christopher Nolan / sci-fi titles.

## 👤 Author

**Akshit Sharma** — B.Tech Engineering Student · Machine Learning & Data Science Enthusiast

- GitHub: https://github.com/akshitsharma009
- LinkedIn: https://www.linkedin.com/in/akshit-sharma-7427362a0

## 🏁 Conclusion

This project demonstrates a practical application of NLP and similarity-based machine learning in a
real-world recommendation system — from data processing to live deployment.
