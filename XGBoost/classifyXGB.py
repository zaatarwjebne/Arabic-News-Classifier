import tkinter as tk
from tkinter import ttk
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
import pyarabic as araby

root = tk.Tk()
root.title("Arabic News Article Categorization")

# Load the saved vectorizer and trained classifier from .pkl files
vectorizer = joblib.load("Models/XGBoost/XGBoost_tfidf_vectorizer.pkl")
xgb_model = joblib.load("Models/XGBoost/XGBoost_model.pkl")

stemmer = ISRIStemmer()
tokenizer = nltk.RegexpTokenizer(r"\w+")

def remove_stop_words(text):
    arabic_stopwords = set(stopwords.words('arabic'))
    words = text.split()
    filtered_words = [word for word in words if word not in arabic_stopwords]
    return ' '.join(filtered_words)

def arabic_stemmer(text):
        Arabic_Stemmer = ISRIStemmer()
        text = [Arabic_Stemmer.stem(y) for y in text.split()]
        return " ".join(text)

def normalize_arabic(text):
    text = text.strip()
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("[إأٱآا]", "ا", text)
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('ييي', 'ي')
    text = text.replace('اا', 'ا')
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'(.)\1+', r"\1\1", text) 
    text = araby.strip_tashkeel(text)
    text = araby.strip_diacritics(text)
    text=''.join([i for i in text if not i.isdigit()])
    return text

def clean_text(text):
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    text = re.sub('[A-Za-z]+',' ',text)
    text = ''.join([i for i in text if not i.isdigit()])
    return text.strip()

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_extra_Space(text):
    text = re.sub('\s+', ' ', text)
    return  " ".join(text.split())

def tokenize_categories(text):
    text = remove_urls(text)
    text = clean_text(text)
    text = remove_stop_words(text)
    text = normalize_arabic(text)
    text = remove_extra_Space(text)
    text = arabic_stemmer(text)
    return text


def predict_category():
    input_text = input_textbox.get("1.0", "end-1c")
    if input_text:
        preprocessed_text = tokenize_categories(input_text)
        vectorized_text = vectorizer.transform([preprocessed_text])
        predicted_prob = xgb_model.predict_proba(vectorized_text)[0]
        predicted_category = xgb_model.predict(vectorized_text)[0]
        result_label.config(text=f"Predicted category: {predicted_category}, Confidence: {predicted_prob.max():.2%}")
    else:
        result_label.config(text="Please enter some text.")


input_textbox = tk.Text(root, height=10)
input_textbox.pack(pady=10)

submit_button = ttk.Button(root, text="Submit", command=predict_category)
submit_button.pack(pady=10)

result_label = ttk.Label(root, text="")
result_label.pack()

root.mainloop()