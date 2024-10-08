import tkinter as tk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
import pyarabic as araby

vectorizer = joblib.load("Models/KNN/tfidf.pkl")
model = joblib.load("Models/KNN/KNN_model.pkl")

root = tk.Tk()
root.geometry("400x200")
root.title("Arabic News Classifier")
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
    article = article_entry.get("1.0", tk.END)
    article = tokenize_categories(article)
    article = vectorizer.transform([article])
    prediction = model.predict(article)[0]
    confidence = model.predict_proba(article).max() * 100
    result_label.config(text=f"Predicted category: {prediction}\nConfidence: {confidence:.2f}%")

article_entry = tk.Text(root, height = 10,width=50)
article_entry.pack(pady=10)
predict_button = tk.Button(root, text="Predict", command=predict_category)
predict_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()