# Import necessary libraries
import fitz  # PyMuPDF for PDF extraction
import pandas as pd
import re
import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Define paths
pdf_path = ''
csv_path = 'sunan_an_nasai.csv'
cleaned_csv_path = 'sunan_an_nasai_cleaned.csv'
model_save_path = "./distilbert-finetuned"

# Step 1: Extract text from PDF and save to CSV
def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    page_data = [(page_num + 1, page.get_text()) for page_num, page in enumerate(pdf_document)]
    pdf_document.close()
    return pd.DataFrame(page_data, columns=['Page', 'Content'])

# Save extracted data to CSV
df = extract_text_from_pdf(pdf_path)
df.to_csv(csv_path, index=False)
print(f"CSV file saved at: {csv_path}")

# Step 2: Clean CSV data by removing special characters
df['Content'] = df['Content'].str.replace('\n', ' ').str.replace('\r', '').str.replace('"', '')
df.to_csv(cleaned_csv_path, index=False)
print(f"Cleaned CSV file saved at: {cleaned_csv_path}")

# Step 3: Load and prepare data for Question Answering
df = pd.read_csv(cleaned_csv_path)

# Initialize tokenizer and model for Question Answering
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Define function to tokenize data
def tokenize_data(examples):
    questions = examples['Questions'].tolist()
    contexts = examples['Content'].tolist()
    encodings = tokenizer(questions, contexts, truncation=True, padding='max_length', return_tensors='pt')
    encodings['start_positions'] = torch.zeros(len(contexts), dtype=torch.long)
    encodings['end_positions'] = torch.zeros(len(contexts), dtype=torch.long)
    return encodings

# Split and tokenize data for training
train_data, val_data = train_test_split(df, test_size=0.1)
train_encodings = tokenize_data(train_data)
val_encodings = tokenize_data(val_data)

# Define a custom Dataset class
class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = QADataset(train_encodings)
val_dataset = QADataset(val_encodings)

# Step 4: Set up training arguments and train model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print("Model and tokenizer saved.")

# Step 5: Extract Hadiths based on chapter title
def find_hadiths_by_chapter(chapter_title, pdf_text):
    pattern = re.compile(rf'{chapter_title}\s*\.+(.*?)(?=\n\d+\s*\.)', re.DOTALL)
    matches = pattern.findall(pdf_text)
    relevant_hadiths = [line.strip() for line in matches if line.strip() and not re.search(r'[\u0600-\u06FF]', line)]
    return relevant_hadiths[:4]

# Load and search for Hadiths in PDF
pdf_text = extract_text_from_pdf(pdf_path)
chapter_title = "Cleaning Oneself With Water"
found_hadiths = find_hadiths_by_chapter(chapter_title, pdf_text)

# Display Hadiths
if found_hadiths:
    print(f"Relevant Hadiths about '{chapter_title}':")
    for hadith in found_hadiths:
        print(hadith.strip())
else:
    print(f"No relevant Hadiths found about '{chapter_title}'.")
