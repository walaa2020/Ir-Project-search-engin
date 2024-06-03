import csv
import math
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import pandas as pd
from text_processing import get_preprocessed_text_terms


def read_dataset_and_process(file_path:str) -> list:
    items = []
    number=[]
    document_process=[]
    normal_text=[]
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        line_count = 0
        for row in reader:
            if line_count >= 1 and line_count <= 100000:
                items.append(row)
                num = row[0]
            
                text = row[1]  
            
              
                if isinstance(text, str):
                    number.append(num)
                    normal_text.append(text)
                    pro=get_preprocessed_text_terms(text)
                    document_process.append(pro)
               
                

            line_count += 1
            if line_count > 100000:
                break
    return normal_text,document_process,number





def get_corpus(file_path, limit=100000):
    corpus = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            doc_id = row[0]
            doc_text = row[1]
            corpus[doc_id] = doc_text
    return corpus


# inverted index
def get_inverted_index(corpus):
    inverted_index = defaultdict(list)
    for doc_id, doc_content in corpus.items():
        terms = get_preprocessed_text_terms(doc_content)
        unique_terms = set(terms)
        for term in unique_terms:
            inverted_index[term].append(doc_id)
    return dict(inverted_index),inverted_index


def write_in_file(file_name,value):
    
    with open(file_name, "wb") as file:
    
     pickle.dump(value, file)

def read_in_file(file_name):
    
    with open(file_name, "rb") as file:
     value=pickle.load(file)
    
    return value



def calculate_tf(doc):
    tf = {}
    terms = get_preprocessed_text_terms(doc)
    term_count = len(terms)
    for term in terms:
        tf[term] = terms.count(term) / term_count
    return tf

def calculate_tf_df(corpus:dict):

    tf_results = {}

    for  doc_id,doc_text in corpus.items():

        tf_results[doc_id]= calculate_tf(doc_text)

    tf_df = pd.DataFrame(tf_results)
    return tf_df,tf_results
    


def calculate_idf(inverted_index,corpus):
    idf = {}
    
    docs_count = len(corpus)

    for term, doc_ids in inverted_index.items():
        idf[term] = math.log((docs_count / len(doc_ids)) + 1)
    
    return idf

def calculate_idf_df(inverted_index,corpus):
                    
    idf_df = pd.DataFrame(calculate_idf(inverted_index,corpus), index=["idf"])
    return idf_df

def calculate_tf_idf (document_process_list:list,tf_dict:dict,idf_dict:dict):
    tf_idf={}
    for terms in document_process_list:
        for term in terms:
            if term in tf_dict and term in idf_dict:
                tf_idf[term] = tf_dict[term] * idf_dict[term]
            else:
    
                tf_idf[term] = 0 
    

    return tf_idf


def calculat_tf_idf_df(tf_idf):
    tf_idf_df = pd.DataFrame(tf_idf, index=["tf_idf"])
    return tf_idf_df


def get_vectorize(document_process:list):
    
    vectorizer = TfidfVectorizer(preprocessor=get_preprocessed_text_terms)
    tfidf_matrix = vectorizer.fit_transform(document_process)
    with open("tfidf_matrix1.pkl", "wb") as file:
        pickle.dump(tfidf_matrix, file)
        file.close()
            

    with open("vectorizer1.pkl", "wb") as file:
        pickle.dump(vectorizer, file)
        file.close()
    return vectorizer,tfidf_matrix