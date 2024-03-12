from flask import Flask, request, jsonify
from config import *
from azure_ai_vector_search import *
from open_ai import *
app = Flask(__name__)

list_of_fields = ["content"]

@app.route('/hello' , methods=['GET'])
def Hello():
    return "Hello, World!"

@app.route('/vector_search', methods=['POST'])
def vector_search():
    job_description = request.json['job_description']
    selected_analysis = request.json['selected_analysis']
    if job_description !='':
        results_content = get_search_results(selected_analysis,job_description)
        content = "\n".join(results_content)
        #st.markdown(f' Match Results {content}')
        # get the reply from the LLM
        reply = get_reply(job_description, content)

def get_reply(user_input, content):
    conversation=[{"role": "system", "content": "You are an AI resume assistant. You will be provided with Job Description and resume of candidates applied for the job. Your task is to compare the resume of the candidate with the job description and provide an output in the below format for each candidate.\n{\nName: Candidate Name,\nTop Skills : list the top skills of the candidate identified as per the job description provided,\nAccuracy : Rate the candidate profile in percentage,\nReason: provide the reason why you think the accuracy is\n}\n{Name: Candidate Name,\nTop Skills : list the top skills of the candidate identified as per the job description provided,\nAccuracy : Rate the candidate profile in percentage,\nReason: provide the reason why you think the accuracy is\n}\nIf you are not clear ask follow up questions before generating the output. As this is related to jobs BE VERY CAREFUL with your analysis.\n"}]
    reply = generate_reply_from_context(user_input, content, conversation)
    return reply

def get_search_results(selected_analysis, user_input):
    custom_vector_search = CustomVectorSearch(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        key=AZURE_SEARCH_ADMIN_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME,
        number_results_to_return=NUMBER_OF_RESULTS_TO_RETURN,
        number_near_neighbors=NUMBER_OF_NEAR_NEIGHBORS,
        embedding_field_name = embedding_field_name,
        semantic_config = AZURE_SEARCH_SEMANTIC_CONFIG_NAME)
     
    if selected_analysis == 'Vector Search':
         results_content = \
            custom_vector_search.get_results_vector_search(user_input,list_of_fields)
    elif selected_analysis == 'Hybrid Search':
         results_content = \
            custom_vector_search.get_results_hybrid_search(user_input,list_of_fields)
    elif selected_analysis == 'Exhaustive KNN Search':
            results_content = \
                custom_vector_search.get_results_exhaustive_knn(user_input,list_of_fields)
    elif selected_analysis == 'Semantic Search':
            results_content = \
                custom_vector_search.get_results_vector_search(user_input,list_of_fields)
                
    return results_content
if __name__ == '__main__':
    app.run(debug=True)