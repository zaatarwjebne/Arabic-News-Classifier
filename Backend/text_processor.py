import os
import re
import pyarabic.araby as araby
from nltk.stem import ISRIStemmer


# Initialize the Arabic stemmer
stemmer = ISRIStemmer()

def keep_arabic_and_whitespace(text):
    """
    Keep only Arabic characters and whitespace
    Arabic Unicode ranges:
    - Basic Arabic: \u0600-\u06FF
    - Arabic Supplement: \u0750-\u077F
    - Arabic Extended-A: \u08A0-\u08FF
    - Arabic Presentation Forms-A: \uFB50-\uFDFF
    - Arabic Presentation Forms-B: \uFE70-\uFEFF
    """
    # Define Arabic Unicode pattern including all Arabic ranges
    arabic_pattern = r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]'

    # Remove any character that doesn't match the Arabic pattern
    text = re.sub(arabic_pattern, ' ', str(text))

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

def remove_english_and_numbers(text):
    """Remove English characters and numbers"""
    # Remove English letters and numbers
    text = re.sub(r'[a-zA-Z0-9]+', ' ', text)
    return text


def remove_arabic_punctuations(text):
    """Remove Arabic punctuation marks and symbols"""
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ'''
    english_punctuations = '`÷×<>_()*&^%][،/:"\'}{~¦+|!"…"–'
    all_punctuations = set(arabic_punctuations + english_punctuations)

    # Remove punctuations
    text = ''.join([char for char in text if char not in all_punctuations])
    return text


def remove_repeating_char(text):
    """Remove repeating characters keeping only one occurrence"""
    return re.sub(r'(.)\1+', r'\1', text)


def remove_urls(text):
    """Remove URLs from text"""
    url_pattern = re.compile(
        r'https?://\S+|www\.\S+|\S+\.com\S*|\S+\.org\S*|\S+\.edu\S*|\S+\.gov\S*|\S+\.net\S*'
    )
    return url_pattern.sub(' ', text)


def remove_emails(text):
    """Remove email addresses from text"""
    email_pattern = re.compile(r'\S+@\S+')
    return email_pattern.sub(' ', text)


def remove_hashtags_mentions(text):
    """Remove hashtags and mentions"""
    # Remove hashtags and mentions (Arabic and English)
    text = re.sub(r'#\w+|@\w+', ' ', text)
    return text



def remove_stop_words(text):
    """Remove Arabic stop words"""
    # Get the absolute path of the current file (datafiler.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the stop words file
    stop_words_path = os.path.join(current_dir, 'resources/stop_words_arabic.txt')
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        arabic_stopwords = f.read().splitlines()
    words = text.split()
    words = [word for word in words if word not in arabic_stopwords]
    return ' '.join(words)

def stem_words(text):
    """Stem Arabic words"""
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)


def preprocess_arabic_text(text):
    """Complete Arabic text preprocessing pipeline"""
    # Convert to string if not already
    text = str(text)

    # Remove URLs, emails, hashtags, and mentions first
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_hashtags_mentions(text)

    # Remove English characters and numbers
    text = remove_english_and_numbers(text)

    # Remove diacritics (tashkeel)
    text = araby.strip_tashkeel(text)

    # Remove tatweel (stretching)
    text = araby.strip_tatweel(text)

    # Normalize hamza
    text = araby.normalize_hamza(text)

    # Keep only Arabic characters and whitespace
    text = keep_arabic_and_whitespace(text)

    # Remove Arabic punctuations
    text = remove_arabic_punctuations(text)

    # Remove repeating characters
    text = remove_repeating_char(text)

    text = remove_stop_words(text)

    text = stem_words(text)
    # Final whitespace cleanup
    text = ' '.join(text.split())

    return text