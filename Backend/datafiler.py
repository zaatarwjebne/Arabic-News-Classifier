import kagglehub
import os
import pandas as pd
from text_processor import preprocess_arabic_text  # Import the preprocessing function



# Download the dataset using kagglehub
dataset_path = kagglehub.dataset_download(
    'haithemhermessi/sanad-dataset'
)


# Lists to store all texts and their categories
all_texts = []
all_categories = []

# Process each category
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    
    if os.path.isdir(folder_path):
        print(f"Processing category: {folder_name}")
        
        # Process all files in the category
        for txt_file in os.listdir(folder_path):
            if txt_file.endswith('.txt'):
                txt_path = os.path.join(folder_path, txt_file)
                
                with open(txt_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Apply preprocessing steps
                content = preprocess_arabic_text(content)
                
                all_texts.append(content)
                all_categories.append(folder_name)

# Create DataFrame
df = pd.DataFrame({
    'text': all_texts,
    'category': all_categories
})

# Drop duplicates
df = df.drop_duplicates()

# Remove rows where category is 'tech'
df = df[df['category'] != 'tech']

# Save to a single CSV file
df.to_csv('dataset.csv', index=False, encoding='utf-8-sig')

print(f"Saved dataset with {len(df)} samples to dataset.csv")
print(f"Number of categories: {len(df['category'].unique())}")
print("Categories:", df['category'].unique())