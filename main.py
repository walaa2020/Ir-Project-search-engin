# import pickle
# from ir import read_dataset_and_process,calculate_tf,calculate_tf_df,get_corpus,get_inverted_index,calculate_idf,calculate_idf_df,calculate_tf_idf,calculat_tf_idf_df
# from ir import get_vectorize,read_in_file

# file_path = "D:/Visual Studio Code/documents/wikIR1k/documents.csv"
# # document_process_list=[]
# normal_text_list=[]
# corpus = {}
# tf_list={}
# dataset_name1='wikIR1k'

# normal_text_list,document_process_list,number_list  = read_dataset_and_process(file_path)
# # print(document_process_list)
# # print("/////////////////////////////////")
# corpus=get_corpus(file_path)
# result_tf_df,result_tf=calculate_tf_df(corpus)
# dic,inverted_index=get_inverted_index(corpus)

# result_idf=calculate_idf(inverted_index,corpus)
# result_idf_df=calculate_idf_df(inverted_index,corpus)

# dict_tf = {}

# for doc_id,dict1 in result_tf.items() :
  
#   for term , value in dict1.items():
#     dict_tf[term] = value

# sorted_doc_tf = dict(sorted(dict_tf.items()))
# sorted_idf = dict(sorted(result_idf.items()))

# result_tf_idf=calculate_tf_idf(normal_text_list,sorted_doc_tf,sorted_idf)


# tf_idf_df=calculat_tf_idf_df(result_tf_idf)

# print("done1")
# vectonizer1,tfidf_matrix1 = get_vectorize(document_process_list)
# result1 = read_in_file("D:\\Visual Studio Code\\Python_Projects\\IR__Project\\tfidf_matrix1.pkl")
# print (result1)
# print ("//////////////////////////////////////////////////")

# result2 = read_in_file("D:\\Visual Studio Code\\Python_Projects\\IR__Project\\vectorizer1.pkl")
# print (result2)

# print("done2")


