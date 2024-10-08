import tkinter as tk
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tkinter import Tk, Label, Text, Button, Menu
import araby

# Load the Word2Vec model and CNN model
w2v_model = Word2Vec.load('Models/word2vec.model')
model = load_model('Models/cnn_model.keras')
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
    article = text_box.get("1.0", tk.END)
    tokens = tokenize_categories(article)
    sequence_padded = pad_sequences([tokens], maxlen=3843, padding="post")
    prediction = model.predict(sequence_padded)
    category = np.argmax(prediction)
    confidence = np.max(prediction)
    label = ['Medical', 'Finance', 'Politics', 'Sports', 'Culture', 'Religion'][category]
    result_label.config(text=f"Predicted Category: {label}\nConfidence Score: {confidence:.2f}")

root = tk.Tk()
root.title("Arabic News Article Classifier")

def copy():
    root.clipboard_clear()
    root.clipboard_append(text_box.selection_get())

def paste():
    text_box.insert(tk.INSERT, root.clipboard_get())

menu = Menu(root, tearoff=0)
menu.add_command(label="Copy", command=copy)
menu.add_command(label="Paste", command=paste)

label = Label(root, text="Enter the text of the article:")
label.pack()

text_box = Text(root, height=10, width=50)
text_box.pack()

def show_menu(event):
    menu.post(event.x_root, event.y_root)

text_box.bind("<Button-1>", show_menu)

button = Button(root, text="Predict", command=predict_category)
button.pack()

result_label = Label(root, text="")
result_label.pack()

root.mainloop()