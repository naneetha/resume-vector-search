from config import *
from openai import AzureOpenAI

client = AzureOpenAI(
  azure_endpoint = AZURE_OPENAI_ENDPOINT,
  api_key=AZURE_OPENAI_KEY,  
  api_version="2023-05-15"
)

def create_prompt(context,query):
    header = "You are helpful assistant."
    return header + context + "\n\n" + query + "\n"


def generate_answer(conversation):
    response = client.chat.completions.create(
    model=AZURE_OPENAI_DEPLOYMENT_ID,
    messages=conversation,
    temperature=0,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop = [' END']
    )
    return (response.choices[0].message.content).strip()

def generate_reply_from_context(user_input, content, conversation):
    prompt = create_prompt(content,user_input)            
    conversation.append({"role": "assistant", "content": prompt})
    conversation.append({"role": "user", "content": user_input})
    reply = generate_answer(conversation)
    return reply

# Function to generate embeddings for title and content fields, also used for query embeddings
#text-embedding-ada-002
def generate_embeddings(text_to_embed): 
    print(f' text to embed {text_to_embed} ')  
    # Get the embeddings for the question
    response = client.Embedding.create(input=[text_to_embed], engine='text-embedding-ada-002')
    # Extract the AI output embedding as a list of floats
    embedding = response["data"][0]["embedding"]
    print(f' embeddings are {embedding} created')
    return embedding