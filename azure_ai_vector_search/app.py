import streamlit as st
from pathlib import Path
from azure_ai_vector_store import *
from azure_ai_vector_search import *
st.title("Resume Search Engine with RAG")

st.sidebar.markdown("## Resume Search Engine")
qa_mode = st.sidebar.radio("Select the option", \
                                 ('Upload Resume',
                                  'Resume Matcher'))

list_of_fields = ["content"]

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
##############################################################
 #   "Match results"
###############################################################
if qa_mode == "Resume Matcher" :
     NUMBER_OF_RESULTS_TO_RETURN = st.sidebar.slider("Number of Search Results to Return",\
                                                 1, 10, 3)    
     selected_analysis = st.sidebar.radio("Select the Analysis Type", \
                                 ('Vector Search', 
                                  'Hybrid Search',
                                  'Exhaustive KNN Search',
                                  'Semantic Search'))
     user_input = st.text_area(
    "Job Description",
    "As a Product Design Manager at GitLab, you will be responsible for managing a team of up to 5 talented Product Designers",
    )
     if st.button("Search"):
         if user_input !='':
            results_content = get_search_results(selected_analysis,user_input)
            content = "\n".join(results_content)
            #st.markdown(f' Match Results {content}')
            # get the reply from the LLM
            reply = get_reply(user_input, content)
            st.markdown(f' MATCH RESULTS WITH ACCURACY:\n {reply}')
##############################################################
 #   "Upload Resume"
###############################################################
if qa_mode == "Upload Resume" :
    st.markdown("**Please fill the below form :**")
    with st.form(key="Form :", clear_on_submit = True):
        Name = st.text_input("Name : ")
        Email = st.text_input("Email ID : ")
        File = st.file_uploader(label = "Upload file", type=["txt","pdf","docx"], accept_multiple_files=True)
        Submit = st.form_submit_button(label='Submit')
        

    st.subheader("Details : ")
    st.metric(label = "Name :", value = Name)
    st.metric(label = "Email ID :", value = Email)

    if Submit :  
        upload()
        st.markdown("**The file is sucessfully Uploaded.**")

