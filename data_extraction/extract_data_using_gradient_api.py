#imports
import fitz
import transformers
import torch
import pandas as pd
import os
import fitz  # PyMuPDF
import numpy as np
import re
from dotenv import load_dotenv
from gradientai import Gradient

# Load the environment variables
load_dotenv()

def extract_section(pdf_path, section='Results'):
    # Open the PDF file
    
    regex_pattern = r'(\d+\.\s*)?(Results and Discussion|Results|Discussion)(\s+section)?\b'

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
                 section: str = 'Conclusion'
                 ):
    def get_activation_energy(text):
        gradient = Gradient()
        base_model = gradient.get_base_model(base_model_slug="llama3-8b-chat")
        query = f"If there is a value of activation energy in the following text, please extract it along with its unit. If there is no activation energy or just the equation or 'E' mentioned as activation enery, please type 'null'. If there is value of activation energy and its unit, write only its value along with unit, nothing else.\n\n{text}"
        complete_response = base_model.complete(
        query=query,
    )
        print(f"\n\nExtracted information: {complete_response.generated_output}")
        gradient.close()
        return complete_response.generated_output
    
    #Loading up dataset
    # li_ion             = pd.read_excel('llmroost/datasets/LiIon_exp.xlsx')
    df = pd.read_excel(df_path)

    #Adding a new column to the dataset
    df['activation_energy'] = ''
    
    processed_dois = []
    activation_dict = {}
    #source is the DOIs column
    for i, doi in enumerate(df['source']):
        try:
            print("\n\n")
            mod_doi  = doi.replace('/',' ').replace(':',' ')
            print(mod_doi)
            # other formatting options....
            pdf_path = f'{pdfs_path}/{mod_doi}.pdf'

            # context  = extract_section(pdf_path, regex_pattern=r'(\d+\.\s*)?Experimental(\s+section)?\b')
            section  = extract_section(pdf_path, section=section)
            # Loop over each sentence in the section
            sentences = section.split('. ')
            keyword_found = False
            keyword_string = ""
            response = ""
            for sentence in sentences:
                # Check if the sentence contains the keyword
                if 'activation energy' in sentence or 'activation barrier' in sentence or 'arrhenius' in sentence:
                    keyword_found = True
                    keyword_string += sentence + ". "
            
            # If keyword is found, add the keyword string to the response
            if keyword_found:
                response = keyword_string
            if response:
                response = re.sub(r'[\x00-\x1f]', '', response)
                if doi not in activation_dict:
                    activation_dict[doi] = response
                else:
                    continue
                print(response)
            else:
                print(f"No activation energy found in {doi}")
            processed_dois.append(doi)
        except Exception as e:
            print(e)
            print(f"Error processing {doi}")
            # Open the file in read mode
            with open('failed_dois.txt', 'r') as f:
                # Check if the DOI is already in the file
                if doi not in f.read():
                    # If the DOI is not in the file, open the file in append mode and write the DOI
                    with open('failed_dois.txt', 'a') as f:
                        f.write(doi + '\n')
            continue
    
    # Get the activation energy for each DOI using Gradient API (Llama3-8b model)
    final_dict = {}
    for doi in activation_dict: 
        print(activation_dict[doi])    
        final_info = get_activation_energy(activation_dict[doi])
        final_dict[doi] = final_info
    for doi in final_dict:
        if 'null' not in final_dict[doi]:
            df.loc[df['source'] == doi, 'activation_energy'] = final_dict[doi]
            print(f"Activation energy for {doi}: {final_dict[doi]}")

    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    df.to_excel('datasets/LiIonDatabase.xlsx', index=False)

if __name__ == '__main__':
    extract_data(df_path='LiIonDatabase.xlsx', 
                 pdfs_path='Li-ion-papers', 
                 section='Results')
