#imports
import fitz
import transformers
import torch
import pandas as pd
import os
import fitz  # PyMuPDF
import shutil
import numpy as np
import re

def extract_section(pdf_path, section='Experimental'):
    # Open the PDF file
    if section == 'Experimental':
        regex_pattern = r'(\d+\.\s*)?Experimental(\s+section)?\b'
    elif section == 'Conclusion':
        regex_pattern = r'(\d+\.\s*)?Conclusions?(\s+section)?\b'

    pdf_document = fitz.open(pdf_path)
    section_text = ""
    section_title_re = re.compile(regex_pattern)

    # Loop through each page in the PDF
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        page_text = page.get_text()

        match = section_title_re.search(page_text)
        if match:
            # Check if the match is at the end of the page text or followed by a non-word character
            if match.end() == len(page_text) or not page_text[match.end()].isalnum():
                section_text += page_text.split(match.group(), 1)[1]
                break
        # if section_title_re.search(page_text):
        #     section_text += page_text.split(section_title_re.search(page_text).group(), 1)[1]
        #     break
    pdf_document.close()

    return section_text.strip()

def extract_all_text(pdf_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Initialize empty string to store the text of the section
    all_text = ""

    # Loop through each page in the PDF
    for page_num in range(len(pdf_document)):
        # Get the text of the current page
        page = pdf_document.load_page(page_num)
        page_text = page.get_text()
        all_text += page_text

    # Close the PDF document
    pdf_document.close()

    return all_text.strip()


def extract_data(df_path: str,
                 pdfs_path: str,
                 section: str = 'Conclusion', #or Experimental for now.
                 prompt: str  = ''):
    
    #Loading up dataset
    # li_ion             = pd.read_excel('llmroost/datasets/LiIon_exp.xlsx')
    df       = pd.read_excel(df_path)
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.float16},
        device_map="auto",
        token='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')
    
    if section == 'Conclusion':
        prompt = "Summarize: what are the conclusions?"
    elif section == 'Experimental':
        prompt = "Summarize: what are the experimental conditions?"
    
    # you can experiment with different prompts/conditions here

    #source is the DOIs column
    for i, doi in enumerate(df['source']):
        if not df.loc[df['source'] == doi, f'{section}'].empty:
            if not np.isnan(df.loc[df['source'] == doi, f'{section}'].values[0]):
                continue

    mod_doi  = doi.replace('/','_') #or '/',' ' depending on naming schemes we used
    # mod_doi  = mod_doi.replace(':',' ')
    # other formatting options....
    pdf_path = f'{pdfs_path}/{mod_doi}.pdf'

    # context  = extract_section(pdf_path, regex_pattern=r'(\d+\.\s*)?Experimental(\s+section)?\b')
    context    = extract_section(pdf_path, section=section)

    messages = [
    {"role": "system", "content": context},
    {"role": "user", "content": "Summarize: what are the conclusions?"},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.2,
        top_p=0.9)

    response = outputs[0]["generated_text"][len(prompt):]
    response = response.replace('\x01', '') #eliminating some ASCII characters
    response = response.replace('\x05', '')
    df.loc[df['source'] == doi, '{se}'] = response
    df.to_excel('datasets/LiIonDatabase.xlsx', index=False)

if __name__ == '__main__':
    extract_data(df_path='data_extraction/LiIonDatabase.xlsx', 
                 pdfs_path='data_extraction/Li-ion-data-papers', 
                 section='Conclusion')

