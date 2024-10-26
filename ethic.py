# Import necessary libraries
import fitz  # PyMuPDF for PDF extraction
import pandas as pd
import re
import requests
import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import streamlit as st

# Define paths
pdf_url = 'https://raw.githubusercontent.com/Kaleemyousaf/hadith/main/Sunan%20an-Nasai%20Vol.%201%20-%201-876.pdf'
pdf_path = 'Sunan_an_Nasai_Vol_1_1-876.pdf'  # Local path to save the downloaded PDF
cleaned_csv_path = "cleaned_data.csv"  # Define a path for the cleaned CSV file

# Step 1: Download the PDF file
response = requests.get(pdf_url)
with open(pdf_path, 'wb') as f:
    f.write(response.content)
print(f"PDF file downloaded and saved as: {pdf_path}")

# Step 2: Extract text from PDF and save to CSV
def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    page_data = [(page_num + 1, page.get_text()) for page_num, page in enumerate(pdf_document)]
    pdf_document.close()
    return pd.DataFrame(page_data, columns=['Page', 'Content'])

# Step 3: Clean CSV data by removing special characters
def clean_data(df):
    df['Content'] = df['Content'].str.replace('\n', ' ').str.replace('\r', '').str.replace('"', '')
    return df

# Load and clean PDF data
df = extract_text_from_pdf(pdf_path)
df = clean_data(df)

# Load PDF content for searching
pdf_text = df['Content'].str.cat(sep=' ')  # Combine all content into a single string

# Function to find Hadiths based on the query
def find_hadiths_by_query(query, pdf_text):
    pattern = re.compile(rf'(.*?{re.escape(query)}.*?)(?=\n\d+\s*\.)', re.DOTALL)
    matches = pattern.findall(pdf_text)
    relevant_hadiths = [line.strip() for line in matches if line.strip() and not re.search(r'[\u0600-\u06FF]', line)]
    return relevant_hadiths

# Streamlit Application
st.title("Hadith Search Application")
st.write("Enter your query to search for relevant Hadiths from the PDF.")

# User input for query
user_query = st.text_input("Query:")

# Search button
if st.button("Search"):
    if user_query:
        found_hadiths = find_hadiths_by_query(user_query, pdf_text)

        # Display results
        if found_hadiths:
            st.write(f"Relevant Hadiths for your query '{user_query}':")
            for hadith in found_hadiths:
                st.write(hadith.strip())
        else:
            st.write(f"No relevant Hadiths found for your query '{user_query}'.")
    else:
        st.write("Please enter a query to search.")

# Run this app with: streamlit run your_script_name.py
