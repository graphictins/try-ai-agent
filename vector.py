
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("data/burgers.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"

add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids       = []

    for index, row in df.iterrows() :
        document = Document(
            page_content = "",
            metadata = { 
                "name"  : row["name"] ,
                "price" : row["price"],
                "type"  : row["type"] ,
            },
            id = str(index)
        )

        ids.append(str(index))
        documents.append(document)

vector_store = Chroma(
    collection_name    = "data_stuff",
    persist_directory  = db_location ,
    embedding_function = embeddings  ,
)

if add_documents : 
    vector_store.add_documents( documents=documents, ids=ids )

retriever = vector_store.as_retriever(
    search_kwargs = { "k" : 8 }
)
