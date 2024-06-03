import csv
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from text_processing import get_preprocessed_text_terms
from main import number_list
from ir import read_in_file 


def read_queries_and_process(file_path: str) -> list:
    items = []
    number = []
    queries_process = []
    normal_queries = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            items.append(row)
            num = row[0]
            text = row[1]
            if isinstance(text, str):
                number.append(num)
                normal_queries.append(text)
                pro = get_preprocessed_text_terms(text)
                queries_process.append(pro)
    return normal_queries, queries_process

def get_corpus(file_path):
    corpus = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            doc_id = row[0]
            doc_text = row[1]
            corpus[doc_id] = doc_text
    return corpus

def create_query_vector(query:str,i:int,vectonizer1):
  print("query: ",query)

  tfidf_matrix = vectonizer1.transform([query])
  file_path = f"queries/tfidf_matrix1_query_{i}.pkl"

  with open(file_path, "wb") as file:
    pickle.dump(tfidf_matrix, file)
    file.close()
            
  return tfidf_matrix

def search(query_vector: str,vectore_matrix,doc_ids):
  similarity_threshold=0.05
  similarity_matrix = cosine_similarity(query_vector,vectore_matrix)
  document_ranking = dict(zip(doc_ids, similarity_matrix.flatten()))
  filtered_documents = {key: value for key, value in document_ranking.items() if value >= similarity_threshold}
  sorted_dict = sorted(filtered_documents.items(), key=lambda item: item[1], reverse=True)
  return  sorted_dict

def get_queries_answers(queries_processed ):
  queries = queries_processed
  queries_answers={}
  vectore_matrix=read_in_file("D:\\Visual Studio Code\\Python_Projects\\IR__Project\\tfidf_matrix1.pkl")
  vectorizer = read_in_file("D:\\Visual Studio Code\\Python_Projects\\IR__Project\\vectorizer1.pkl")
  doc_ids = number_list

  i = 1
  for id,query in list(queries.items()):
    print("your question is: " + id)
    query_vector=create_query_vector(query,i, vectorizer)
    top_related_docs = search(query_vector,vectore_matrix,doc_ids)
    queries_answers[id]=top_related_docs
    print(len(top_related_docs))
    i+=1
    
    return queries_answers

def criteria_results(ranked_docs, qrels, k=None):
  metrics = {}
  ap_sum = 0
  mrr_sum = 0
  p10_sum = 0
  overall_precision = 0
  overall_recall = 0
  overall_f1_score = 0
  for query_id in ranked_docs.keys():
    ranked_list = ranked_docs[query_id]
    relevant_docs = [doc_id for doc_id, score in qrels[query_id] if score > 0]
    if k is not None:
      ranked_list = {key: value for key, value in list(ranked_list)[:k]}
    tp = len(set(ranked_list).intersection(set(relevant_docs)))
    precision = tp / len(ranked_list) if len(ranked_list) > 0 else 0
    recall = tp / len(relevant_docs) if len(relevant_docs) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    overall_precision += precision
    overall_recall += recall
    overall_f1_score += f1_score
    ap = 0
    relevant_docs_seen = set()
    for i, doc_id in enumerate(ranked_list):
      if doc_id in relevant_docs and doc_id not in relevant_docs_seen:
        ap += (len(relevant_docs_seen) + 1) / (i + 1)
        relevant_docs_seen.add(doc_id)
        if len(relevant_docs_seen) == 1:
          mrr_sum += 1 / (i + 1)
        if len(relevant_docs_seen) == len(relevant_docs):
          break
    ap /= len(relevant_docs) if len(relevant_docs) > 0 else 1
    ap_sum += ap
    p10 = len(set(list(tuple(ranked_list))[:10]).intersection(set(relevant_docs)))
    p10_sum += p10 / 10
    metrics[query_id] = {
      'precision': precision,
      'recall': recall,
      'f1_score': f1_score,
      'ap': ap,
      'p10': p10
    }
  overall_precision = overall_precision / len(ranked_docs)
  overall_recall = overall_recall / len(ranked_docs)
  overall_f1_score = overall_f1_score / len(ranked_docs)
  overall_ap = ap_sum / len(ranked_docs)
  overall_mrr = mrr_sum / len(ranked_docs)
  overall_p10 = p10_sum / len(ranked_docs)
  metrics['overall'] = {
    'precision': overall_precision,
    'recall': overall_recall,
    'f1_score': overall_f1_score,
    'map': overall_ap,
    'mrr': overall_mrr,
    'p10': overall_p10
  }
  # return metrics
  return metrics["overall"]


