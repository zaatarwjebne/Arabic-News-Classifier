import customtkinter
import os
from tkinter import messagebox
customtkinter.set_appearance_mode('light')
customtkinter.set_default_color_theme('blue')
from gensim.models import Word2Vec
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from tkinter import Menu
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tkinter import Menu
import pyarabic.araby as araby
 
class Preprocessor:
    @staticmethod
    def remove_stop_words(text):
        nltk.download('stopwords')  # Download the stopwords corpus if not already downloaded
        arabic_stopwords = set(stopwords.words('arabic'))  # Load Arabic stopwords
        words = text.split()
        filtered_words = [word for word in words if word not in arabic_stopwords]
        return ' '.join(filtered_words)

    @staticmethod
    def arabic_stemmer(text):
        Arabic_Stemmer = ISRIStemmer()
        text = [Arabic_Stemmer.stem(y) for y in text.split()]
        return " ".join(text)

    @staticmethod
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

    @staticmethod
    def remove_non_arabic(text):
        text = re.sub('[A-Za-z]+',' ',text)
        return text

    @staticmethod
    def remove_numbers(text):
        text=''.join([i for i in text if not i.isdigit()])
        return text

    @staticmethod
    def remove_punctuations(text):
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛',"", )
        text = re.sub('\s+', ' ', text)
        text =  " ".join(text.split())
        return text.strip()

    @staticmethod
    def remove_urls(text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    @staticmethod
    def remove_extra_Space(text):
        text = re.sub(r'\s+', ' ', text)  # remove extra spaces
        text = re.sub(r'\n+', '\n', text)  # remove extra newline characters
        text = re.sub(r'\n', ' ', text)  # replace newline characters with spaces
        return " ".join(text.split())
    @staticmethod
    def preprocess_Text(text):
        text = Preprocessor.remove_urls(text)
        text = Preprocessor.remove_punctuations(text)
        text = Preprocessor.remove_numbers(text)
        text = Preprocessor.remove_non_arabic(text)
        text = Preprocessor.remove_stop_words(text)
        text = Preprocessor.normalize_arabic(text)
        text = Preprocessor.remove_extra_Space(text)
        text = Preprocessor.arabic_stemmer(text)
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        text = tokenizer.tokenize(text)
        return text
    
class MainGUI():

    def predict_category():
        ## WARNING: MAINLY FOR CNN, USE YOU OWN CLASSIFICATION TECHNIQUE.
        if Selected == 1:
            cnnmodel = load_model('Models/cnn_model.keras')
            embedding = Word2Vec.load('Models/word2vec.model')
            article = Input_Textbox.get("1.0", "end")
            tokens = Preprocessor.preprocess_Text(article)
            sequence = [embedding.wv.key_to_index[token] if token in embedding.wv.key_to_index else 0 for token in tokens]
            sequence_padded = pad_sequences([sequence], maxlen=3843, padding="post")
            prediction = cnnmodel.predict(sequence_padded)
            category = np.argmax(prediction)
            confidence = np.max(prediction)
            label = ['Medical', 'Finance', 'Politics', 'Sports', 'Culture', 'Religion'][category]
            result_label = customtkinter.CTkLabel(Main, text="", font=("Pacifico", 18, "bold"))
            result_label.configure(text=f"")
            result_label.configure(text=f"Predicted Category: {label}\nConfidence Score: {confidence:.2f}")
            result_label.place(x=Main.winfo_screenwidth()/2 - 120,y=Main.winfo_screenheight()/2 - 200, anchor="center")
            
        if Selected == 2:
            xgb_vec = joblib.load("Models/XGBoost/XGBoost_tfidf_vectorizer.pkl")
            xgb_model = joblib.load("Models/XGBoost/XGBoost_model.pkl")
            def predict_XGBoost():
                input_text = Input_Textbox.get("1.0", "end")
                if input_text:
                    preprocessed_text = Preprocessor.preprocess_Text(input_text)
                    vectorized_text = xgb_vec.transform([preprocessed_text])
                    predicted_prob = xgb_model.predict_proba(vectorized_text)[0]
                    predicted_category = xgb_model.predict(vectorized_text)[0]
                    result_label = customtkinter.CTkLabel(Main, text="", font=("Pacifico", 18, "bold"))
                    result_label.configure(text=f"")
                    result_label.place(x=Main.winfo_screenwidth()/2 - 120,y=Main.winfo_screenheight()/2 - 200, anchor="center")
                    result_label.config(text=f"Predicted category: {predicted_category}, Confidence: {predicted_prob.max():.2%}")
                else:
                    result_label.configure(text="Please enter some text.")
                    result_label.place(x=Main.winfo_screenwidth()/2 - 120,y=Main.winfo_screenheight()/2 - 200, anchor="center")
            predict_XGBoost
                
        if Selected == 3:
            knn_vec = joblib.load("Models/KNN/tfidf.pkl")
            knn_model = joblib.load("Models/KNN/KNN_model.pkl")
            def predict_KNN():
                article = Input_Textbox.get("1.0", "end")
                article = knn_vec.transform([article])
                prediction = knn_model.predict(article)[0]
                probabilities = knn_model.predict_proba(article)[0]
                category_index = np.argmax(probabilities)
                label = ['Medical', 'Finance', 'Politics', 'Sports', 'Culture', 'Religion'][category_index]
                confidence = probabilities[category_index] * 100
                result_label = customtkinter.CTkLabel(Main, text="", font=("Pacifico", 18, "bold"))
                result_label.configure(text=f"")
                result_label.place(x=Main.winfo_screenwidth()/2 - 120,y=Main.winfo_screenheight()/2 - 200, anchor="center")
                result_label.configure(text=f"Predicted Category: {label}\nConfidence Score: {confidence:.2f}")
            predict_KNN()
                
        if Selected == 4:
                
                pass
            
                
    def copy():
        Main.clipboard_append(Input_Textbox.selection_get())

    def paste():
        Input_Textbox.insert(Main.INSERT, Main.clipboard_get())
    
    menu = Menu(customtkinter.CTk(), tearoff=0)
    menu.add_command(label="Copy", command=copy)
    menu.add_command(label="Paste", command=paste) 
          
    def destroy_all():
        widgets = Main.winfo_children()
        for widget in widgets:
            if hasattr(widget, 'delete'):
                widget.delete(0, 'end')
            elif hasattr(widget, 'destroy'):
                widget.destroy()
                
    def Continue():
        MainGUI.destroy_all()
        MainGUI.train_test_form()
    
    def reset_window():
        Main.destroy()
        os.startfile(r"MainGUI.py")
    
    def clear_test():
        Input_Textbox.delete('1.0', 'end')

    def go_test():
        global classifier
        global Selected
        classifier = classifierComboBox.get()
        
        if classifier == "Convolutional Neural Networks":
            Selected = 1
            print("CNN")
            MainGUI.create_classifier_form()
            
        elif classifier == "XGBoost":
            Selected = 2
            print("XGBoost")
            MainGUI.create_classifier_form()
            
        elif classifier == "K-Nearest Neighbors":
            Selected = 3
            print("KNN")
            MainGUI.create_classifier_form()
            
        else:
            messagebox.showinfo("Error", "Please choose a classifier")

    def test_classifier_page():
        MainGUI.destroy_all()
        allClassifiers = ["Convolutional Neural Networks", "XGBoost", "K-Nearest Neighbors"]
        ChooseClassLbl = customtkinter.CTkLabel(Main, text="Choose a Classifier for Testing", font=("Pacifico", 40, "bold"))
        global classifierComboBox
        classifierComboBox = customtkinter.CTkComboBox(Main, values=allClassifiers, width=400, height=55, font=("Pacifico", 30, "bold"))
        ConfirmButton = customtkinter.CTkButton(Main, text="Confirm", command=lambda: MainGUI.go_test(), width=100, height=50, font=("Pacifico", 30, "bold"), fg_color="darkblue", hover_color="blue")
        QuitButton = customtkinter.CTkButton(Main, text="Quit", command=quit, width=100, height=50, font=("Pacifico", 30, "bold"), fg_color="darkblue", hover_color="blue")

        ChooseClassLbl.place(relx=0.5, rely=0.1, anchor="center")
        classifierComboBox.place(relx=0.5, rely=0.5, anchor="center")
        ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")
        QuitButton.place(relx=0.5, rely=0.7, anchor="center")


    def create_classifier_form():
        MainGUI.destroy_all()
        ChooseFileLabel = customtkinter.CTkLabel(Main, text="Input An Article", font=("Pacifico", 40, "bold"))
        classifierLabel = customtkinter.CTkLabel(Main, text=classifier, font=("Pacifico", 10, "bold"))
        Classify_Button = customtkinter.CTkButton(Main, text="Classify",width=200, height=62, font=("Pacifico", 30, "bold"), fg_color="darkblue", hover_color="blue", command=lambda:MainGUI.predict_category())
        Clear_Btn = customtkinter.CTkButton(Main, text="Clear",width=180, height=62, font=("Pacifico", 30, "bold"), fg_color="darkblue", hover_color="blue", command=lambda:MainGUI.clear_test())

        global Input_Textbox
        Input_Textbox = customtkinter.CTkTextbox(Main, width=600, height=300, font=("Pacifico", 20, "bold"))

      # Place the elements using pack
        ChooseFileLabel.pack(pady=(20, 10))
        Input_Textbox.pack(pady=10)
        Classify_Button.pack(pady=10)
        Clear_Btn.pack(pady=10)
        classifierLabel.pack(padx=(0, 20))

        # Remove place layout
        ChooseFileLabel.place_forget()
        Input_Textbox.place_forget()
        Classify_Button.place_forget()
        Clear_Btn.place_forget()
        classifierLabel.place_forget()

Main = customtkinter.CTk()
Main.title("Arabic News Classifier")
Main.attributes("-topmost", True)

ScreenWidth = Main.winfo_screenwidth()
ScreenHeight = Main.winfo_screenheight()
Main.geometry("1000x580".format(ScreenWidth, ScreenHeight))

WelcomeLabel = customtkinter.CTkLabel(Main, text="Welcome to the\nMain Page", font=("Pacifico", 40, "bold"))
WelcomeLabel.pack(pady=20)

button_frame = customtkinter.CTkFrame(Main)
button_frame.pack(pady=20)

ContinueButton = customtkinter.CTkButton(button_frame, text="Start", command=lambda: MainGUI.test_classifier_page(),  width=200, height=62, font=("Pacifico", 30, "bold"), fg_color="darkblue", hover_color="blue")
ContinueButton.pack(padx=10, pady=10)

QuitButton = customtkinter.CTkButton(button_frame, text="Quit", command=quit, width=200, height=62, font=("Pacifico", 30, "bold"), fg_color="darkblue", hover_color="blue")
QuitButton.pack(padx=10, pady=10)

button_frame.place(relx=0.5, rely=0.7, anchor="center")
Main.mainloop() 