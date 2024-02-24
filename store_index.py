from src.helper import load_pdf, download_hugging_face_embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
# text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
from pinecone import Pinecone

pc = Pinecone(api_key="c727e84c-88ca-4752-aea8-a70fbedf5980")
index = pc.Index("medical-chatbot")

index_name = "medical-chatbot"

#Creating Embeddings for Each of The Text Chunks & storing
from tqdm.auto import tqdm
from uuid import uuid4

batch_limit = 100

texts = []
metadatas = []

# Assuming 'data' is the list of Document objects
for i, document in enumerate(tqdm(extracted_data)):
    # Extract metadata from the document
    metadata = {
        'source': document.metadata['source'],
        'page no': document.metadata['page']
    }
    # Extract text content from the document
    text = document.page_content  # Assuming 'page_content' contains the text content
    # Now we create chunks from the text content
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    record_texts = text_splitter.split_text(text)
    # Create individual metadata dicts for each chunk
    record_metadatas = [{
        "chunk": j, "text": text, **metadata
    } for j, text in enumerate(record_texts)]
    # Append these to current batches
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)
    # If we have reached the batch_limit we can add texts
    if len(texts) >= batch_limit:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embeddings.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        texts = []
        metadatas = []

# Process the remaining texts if any
if len(texts) > 0:
    ids = [str(uuid4()) for _ in range(len(texts))]
    embeds = embeddings.embed_documents(texts)
    index.upsert(vectors=zip(ids, embeds, metadatas))
