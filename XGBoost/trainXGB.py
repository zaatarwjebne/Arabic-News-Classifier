import pandas as pd
import os
import nltk
import csv
import re
import pyarabic.araby as araby
from nltk.stem import ISRIStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


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

print ("\nRunning XGBoost Model...")
X = vectorizer.fit_transform(df['Content'])
le = LabelEncoder()
y = le.fit_transform (df['Topic'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(
    colsample_bytree=0.6,  # Lower ratio to avoid overfitting
    subsample=0.6,  # Lower ratio to avoid overfitting
    max_depth=4,  # Lower value to avoid overfitting
    gamma=0.1,  # Larger value to avoid overfitting
    eta=0.1,  # Lower value to avoid overfitting
    min_child_weight=7  # Larger value to avoid overfitting
)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1 score:", f1_score(y_test, y_pred, average="weighted"))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print ("\nSaved TF-IDF and XGBoost weights...")
if not os.path.exists('Models/XGBoost'):
    os.makedirs('Models/XGBoost')

# Save the vectorizer
joblib.dump(vectorizer, "Models/XGBoost/XGBoost_tfidf_vectorizer.pkl")
joblib.dump(xgb_model, "Models/XGBoost/XGBoost_model.pkl")

