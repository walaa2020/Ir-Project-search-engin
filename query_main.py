
from query import read_queries_and_process,get_queries_answers,get_corpus,criteria_results
from ir import write_in_file , read_in_file



# quries_path = "D:/Visual Studio Code/documents/wikIR1k/training/queries.csv"

# answers :dict={}

# normal_queries_list,queries_processed_list=read_queries_and_process(quries_path)
# query_corpus = get_corpus(quries_path)

# answers = get_queries_answers(query_corpus)
# # print(answers)
# write_in_file("D:\\Visual Studio Code\\Python_Projects\\IR__Project\\answers.pkl",answers)
ans = read_in_file("D:\\Visual Studio Code\\Python_Projects\\IR__Project\\answers.pkl")
# print(ans)

