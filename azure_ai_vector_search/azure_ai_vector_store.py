#Import Required Packages
import json
import uuid
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from config import *
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

import openai

openai.api_key = OPENAI_API_KEY
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryCaptionResult,
    QueryAnswerResult,
    SemanticErrorMode,
    SemanticErrorReason,
    SemanticSearchResultsType,
    QueryType,
    VectorizedQuery,
    VectorQuery,
    VectorFilterMode,    
)
from azure.search.documents.indexes.models import (  
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    SemanticPrioritizedFields,
    SemanticField,  
    SearchField,  
    SemanticSearch,
    VectorSearch,  
    HnswAlgorithmConfiguration,
    HnswParameters,  
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    ExhaustiveKnnParameters,
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    SemanticField,  
    SearchField,  
    VectorSearch,  
    HnswParameters,  
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)  



# Configure the vector search configuration  
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="myHnsw",
            kind=VectorSearchAlgorithmKind.HNSW,
            parameters=HnswParameters(
                m=4,
                ef_construction=400,
                ef_search=500,
                metric=VectorSearchAlgorithmMetric.COSINE
            )
        ),
        ExhaustiveKnnAlgorithmConfiguration(
            name="myExhaustiveKnn",
            kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
            parameters=ExhaustiveKnnParameters(
                metric=VectorSearchAlgorithmMetric.COSINE
            )
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnsw",
        ),
        VectorSearchProfile(
            name="myExhaustiveKnnProfile",
            algorithm_configuration_name="myExhaustiveKnn",
        )
    ]
)
semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        content_fields=[SemanticField(field_name="content")],
        keywords_fields=[SemanticField(field_name="filename")]
    )
)
# Create the semantic settings with the configuration
semantic_search = SemanticSearch(configurations=[semantic_config])

#Create Search Index client
client = SearchIndexClient(AZURE_SEARCH_SERVICE_ENDPOINT, AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY))
# Create the index
index_name = "resume"
fields = [
        SimpleField(name="documentId", type=SearchFieldDataType.String, filterable=True, sortable=True, key=True),     
        SearchableField(name="content", type=SearchFieldDataType.String),        
        SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=1536, 
                vector_search_profile_name="myHnswProfile")
    ]

index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search
    )
# Create the search index with the semantic settings
#index = SearchIndex(name=index_name, fields=fields,
 #                   vector_search=vector_search, 
 #                   semantic_search=semantic_search)

result = client.create_or_update_index(index)

print(f' Index with name {result.name} created')

#Chunking the text and convert to embeddings and dump to json
def chunk_text():
    #loader =  TextLoader("docs/Resume1.txt", encoding ="utf-8")
    loader = DirectoryLoader('docs/',glob="**/*.txt",loader_cls=TextLoader, show_progress=True)
    documents =  loader.load()
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=50)
    documents =  text_splitter.split_documents(documents)
    print(f' Text files are chunked and send to embedding')
    #Construct json
    docs = []
    for doc in documents:
        docs.append({"documentId":str(uuid.uuid4()),"content":doc.page_content,"embedding":generate_embeddings(doc.page_content)})
    json_data = json.dumps(docs)
    with open('output/ResumeContent.json', 'w') as f:  
        f.write(json_data)
        print(f'json_data are created in outputfolder')
    
# Function to generate embeddings for title and content fields, also used for query embeddings
#text-embedding-ada-002
def generate_embeddings(text_to_embed): 
    print(f' text to embed {text_to_embed} ')  
    # Get the embeddings for the question
    response = openai.Embedding.create(input=[text_to_embed], engine='text-embedding-ada-002')
    # Extract the AI output embedding as a list of floats
    embedding = response["data"][0]["embedding"]
    print(f' embeddings are {embedding} created')
    return embedding

def upload():
    chunk_text()
    #Upload document with embeddings to index
    with open('output/ResumeContent.json', 'r') as f:  
        documents = json.load(f)  
    search_client = SearchClient(endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, index_name=index_name, credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY))
    result = search_client.upload_documents(documents)
    print(f' embedded documents are uploaded successfully')

