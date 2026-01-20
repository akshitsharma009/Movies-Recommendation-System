## 👤 Author

**Akshit Sharma**  
B.Tech Engineering Student  
Machine Learning & Data Science Enthusiast  

- GitHub: https://github.com/akshitsharma009
- LinkedIn: https://www.linkedin.com/in/<your-linkedin-username>

🎬 Movie Recommendation System

An end-to-end content-based movie recommendation system that suggests movies similar to a user’s favorite choice using natural language processing and cosine similarity. The application is deployed as an interactive web app using Streamlit.

🔗 Live App:
https://movies-recommendation-system-fgg2ekv383qu3xhmrc5gva.streamlit.app/

📌 Project Overview

This project recommends movies based on similarity in:

genres

keywords

cast

director

movie overview

It uses TF-IDF vectorization to convert textual metadata into numerical vectors and computes similarity using cosine similarity.

The system is designed to be fast, intuitive, and production-ready.

⚙️ Tech Stack

Programming Language: Python

Libraries:

Pandas

NumPy

Scikit-learn

Streamlit

Machine Learning:

TF-IDF Vectorization

Cosine Similarity

Deployment:

Streamlit Cloud

Version Control:

Git & GitHub

🧠 How It Works

Movie metadata is cleaned and combined into a single text feature.

TF-IDF converts text into numerical feature vectors.

Cosine similarity measures similarity between movies.

The system finds and ranks the most similar movies.

Results are displayed via an interactive Streamlit UI.

🚀 Features

Content-based movie recommendation

Smart fuzzy matching for movie names

Adjustable number of recommendations

Clean, modern Streamlit UI

CSV download for recommendations

Fully deployed and accessible online

📂 Project Structure
movies-recommendation-system/
│
├── app.py                 # Streamlit application
├── movies.csv             # Dataset
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

▶️ How to Run Locally

Clone the repository:

git clone https://github.com/akshitsharma009/movies-recommendation-system.git


Navigate to the project folder:

cd movies-recommendation-system


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py

📈 Example Use Case

User selects Inception

System analyzes metadata similarities

Recommends movies like Interstellar, The Dark Knight, etc.

👤 Author

Akshit Sharma
B.Tech Engineering Student
Machine Learning & Data Science Enthusiast

GitHub: https://github.com/akshitsharma009

LinkedIn: https://www.linkedin.com/in/akshit-sharma-7427362a0

🏁 Conclusion

This project demonstrates practical application of NLP techniques and machine learning similarity models in building a real-world recommendation system, from data processing to live deployment.
