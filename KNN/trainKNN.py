import os
import re
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import pyarabic.araby as araby



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

archive_folder = 'C:\\Path\\to\\data'
output_csv = 'all_topics_numbered.csv'

with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    
    writer.writerow(['Filename', 'Content', 'Topic'])
    
    texts = []
    categories = []
    
    for folder_name in os.listdir(archive_folder):
        folder_path = os.path.join(archive_folder, folder_name)
        
        if os.path.isdir(folder_path):
            topic = folder_name  
            
            for txt_file in os.listdir(folder_path):
                if txt_file.endswith('.txt'):
                    txt_path = os.path.join(folder_path, txt_file)
                    
                    with open(txt_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    
                        content = remove_urls(content)
                        content = clean_text(content)
                        content = remove_stop_words(content)
                        content = normalize_arabic(content)
                        content = remove_extra_Space(content)
                        content = arabic_stemmer(content)
                        tokenizer = nltk.RegexpTokenizer(r"\w+")
                        content = tokenizer.tokenize(content)
                        
                        writer.writerow([txt_file, content, topic])
                        
                        texts.append(content)
                        categories.append(topic)

print(f"Successfully cataloged all files into {output_csv}")

df = pd.read_csv(output_csv)
# Drop the 'Filename' column
df = df.drop(df.columns[0], axis=1)

df.drop_duplicates(subset='Content', inplace=True)

df = df[df['Topic'] != 'Tech']
vectorizer = TfidfVectorizer()
print ("\nGenerating embeddings using TF-IDF...")
texts = [' '.join(text) for text in texts]
X = vectorizer.fit_transform(texts)

print ("\nRunning KNN Model...")
X_train, X_test, y_train, y_test = train_test_split(X, categories, test_size=0.2, random_state=42)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1score:", f1_score(y_test, y_pred, average="weighted"))

print ("\nSaving TF-IDF and KNN weights...")
if not os.path.exists('Models/KNN'):
    os.makedirs('Models/KNN')


joblib.dump(vectorizer, "Models/KNN/tfidf.pkl")
joblib.dump(model, "Models/KNN/KNN_model.pkl")