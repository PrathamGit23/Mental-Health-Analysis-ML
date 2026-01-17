Mental Health Analysis using Machine Learning

This project demonstrates a machine learning–based system that analyzes user-written text to identify potential mental health patterns using Natural Language Processing (NLP).

The goal is to provide an educational demonstration of how text data can be processed and classified using traditional machine learning techniques, presented through a simple web interface.

What this project does
- Takes text input from the user through a web interface
- Cleans and preprocesses the text
- Converts text into numerical features using TF-IDF
- Uses a trained Logistic Regression model to predict mental health categories
- Displays prediction confidence along with common symptoms and helpful resources

This is an NLP-based machine learning project intended for learning and demonstration purposes.

Why this project matters
Mental health discussions often appear in text form (social media, forums, journals).
Analyzing such text using NLP can help identify patterns and raise awareness.

This project serves as a baseline system that can be extended to:
- Deep learning models
- Larger datasets
- Real-time monitoring or analytics tools

How to run the project locally
```bash
git clone https://github.com/PrathamGit23/Mental-Health-Analysis-ML.git
cd Mental-Health-Analysis-ML
pip install flask scikit-learn pandas nltk joblib
python app.py
Then open your browser and go to:

cpp
Copy code
http://127.0.0.1:5000/

DISCLAIMER
This project is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.
