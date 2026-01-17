# Mental Health Analysis using Machine Learning

This project demonstrates a **machine learning–based system** that analyzes user-written text to identify potential mental health patterns using **Natural Language Processing (NLP)**.

The goal is to understand how textual data can be processed and classified to detect mental health–related patterns through a simple web-based interface.

---

## What this project does

- Accepts text input from users through a web interface  
- Cleans and preprocesses the input text  
- Converts text into numerical features using TF-IDF  
- Uses a trained Logistic Regression model to classify mental health categories  
- Displays prediction confidence along with common symptoms and helpful resources  

This is an **NLP-based machine learning project**, intended for learning and demonstration purposes.

---

## Why this project matters

Mental health discussions often appear in text form on platforms such as forums, journals, and social media.  
Analyzing such text using NLP techniques can help identify patterns and raise awareness.

This project is a **baseline implementation** that can be extended to:
- Larger datasets  
- Deep learning–based NLP models  
- Real-time mental health analysis systems  

---

## DISCLAIMER

This project is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.

---

## How to run the project locally

```bash
git clone https://github.com/PrathamGit23/Mental-Health-Analysis-ML.git
cd Mental-Health-Analysis-ML
pip install flask scikit-learn pandas nltk joblib
python app.py

Open your browser and go to:

http://127.0.0.1:5000/
