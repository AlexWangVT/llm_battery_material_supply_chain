import os
os.environ["USER_AGENT"] = "battery_materials_supply_chain_app" # set up USER_AGENT environment to avoid user_agent warning
import yaml
import torch
import datetime
import requests
import weaviate
import asyncio
import datetime
import accelerate
from uuid import uuid4
import streamlit as st
from bs4 import BeautifulSoup 
import weaviate.classes as wvc
import google.generativeai as genai
from weaviate.classes.query import Filter
from dotenv import load_dotenv
from pymongo import MongoClient
from datasets import Dataset
from langchain.schema import Document
from urllib.parse import urlparse, unquote, urljoin
from langchain_community.vectorstores import Weaviate
from weaviate.connect import ConnectionParams
from langchain_community.document_loaders import WebBaseLoader
from peft import get_peft_model, LoraConfig, TaskType
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFacePipeline 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.mongodb import MongodbLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults 
from langchain_community.document_loaders import UnstructuredURLLoader, UnstructuredPDFLoader, PyPDFLoader
from weaviate.collections.classes.config import CollectionConfig, Property, DataType, VectorizerConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline


load_dotenv(dotenv_path="../.env")
user_agent = os.getenv("USER_AGENT", "battery_materials_supply_chain_app")

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

google_api_key = config["api"]["google_api"]
llama_api_key = config["api"]["llama_api"]
local_data_sources = config["database"]["local"]
mongoclient = MongoClient(config["database"]["mongodb_server"])
mongo_db = mongoclient[config["database"]["mongodb_database"]]
mycol = mongo_db[config["database"]["mongodb_collection_name"]]
weaviate_client = weaviate.connect_to_local() # connect to local temporarily, will connect to cloud server later
weaviate_collection_name = config["database"]["weaviate_collection_name"] # Collection name must have the first letter capitalized in Weaviate
weaviate_similarity_factor = config["database"]["weaviate_similarity_factor"]
mongodb_update_start_time = datetime.datetime(2025, 1, 1)
mongodb_update_end_time = datetime.datetime(2025, 5, 15)
text_embedding_model = config["model"]["text_embedding_model"]
text_embedding_model_kwargs = config["model"]["text_embedding_model_kwargs"]
gemini_model_name = config["model"]["gemini_model"]
llama_model_name = config["model"]["llama_model"]


def crawl_nested_tabs(url, base_url, visited, max_depth, depth=0):
    if url in visited or depth > max_depth:
        return []

    visited.add(url)

    try:

        headers = {
            "User-Agent": user_agent
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
    except:
        return []
    
    results = [(url)]

    # Find nested links (e.g., subtabs, buttons, inner menus)
    links = [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True)]

    for link in links:
        if base_url in link:  # Stay within the domain
            results += crawl_nested_tabs(link, base_url, visited, max_depth, depth + 1)

    return results


def check_pdf_extension(file_or_link):
        """
        Checks if a file or link ends with '.pdf'.

        Args:
            file_or_link: The file name or link (string).

        Returns:
            True if the file or link ends with '.pdf', False otherwise.
        """
        return file_or_link.lower().endswith(".pdf")


def mongodb(url, doc, metadata=None):
    def generate_title_from_url(url: str) -> str:
        path = urlparse(url).path
        title = path.strip("/").split("/")[-1].replace("_", " ")
        return unquote(title).title()

    def generate_title_from_filename(url: str) -> str:
        name = os.path.basename(url).replace("_", " ").replace("-", " ")
        return os.path.splitext(name)[0].title()

    if check_pdf_extension(url):
        pdf_document = {
                "content": doc,
                "metadata": {
                    **(metadata or {}),
                    "doc_type": "pdf",
                    "source": url,
                    "title": generate_title_from_filename(url),
                    "timestamp": datetime.datetime.now(datetime.UTC)
                }
            }
        # insert pdf into mongodb
        existing_doc = mycol.find_one({"metadata.source": pdf_document["metadata"]["source"]})
        if existing_doc:
            if existing_doc.get("content", "") != pdf_document['content']:
                mycol.replace_one({"_id": existing_doc["_id"]}, pdf_document)
                print("documents updated!")
            else:
                print("document exists!")
        else:
            mycol.insert_one(pdf_document)
       
    else:
        # insert web articles into mongodb
        insert_docu = {
            "content": doc.page_content,
            "metadata": {
                "source": url,
                "title": generate_title_from_url(url),
                "doc_type": "webpage",
                "timestamp": datetime.datetime.now(datetime.UTC)
            }
        }
        existing_doc = mycol.find_one({"metadata.source": insert_docu["metadata"]["source"]})
        if existing_doc:
            if existing_doc.get("content", "") != insert_docu['content']:
                mycol.replace_one({"_id": existing_doc["_id"]}, insert_docu)
                print("documents updated!")
            else:
                print("document exists!")
        else:
            mycol.insert_one(insert_docu)


def search_load_data(local_data_sources):
    with open(local_data_sources, "r") as file:
        lines = file.readlines()
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent})
        for line in lines:
            if not check_pdf_extension(line): # Load web articles
                visited = set()
                max_depth = 2  # Prevents infinite loops
                all_tabs = crawl_nested_tabs(line, line, visited, max_depth)
                for tab in all_tabs:
                    try: 
                        loader = WebBaseLoader(line, requests_session=session)
                        doc = loader.load()
                        mongodb(tab, doc)
                    except Exception as e:
                        print(f"Error loading {tab}: {e}")
            else:
                loader = PyPDFLoader(line, requests_session=session) # Load pdf files local or online
                async def load_pages():
                    pages = []
                    async for page in loader.alazy_load():
                        pages.append(page)
                    return pages
                pages = asyncio.run(load_pages())
                doc = "\n".join([p.page_content for p in pages])
                metadata = pages[0].metadata if pages else {}
                mongodb(line, doc, metadata)


def text_embedding_model_():
    model_name = text_embedding_model
    model_kwargs = text_embedding_model_kwargs
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return hf


def text_split_embedding():
    def data_retrieval_by_time_mongodb(client):
        db_name = client[mongo_db]
        col_name = db_name[mycol]
        data = col_name.find({"metadata.timestamp": {
                "$gte": mongodb_update_start_time,
                "$lte": mongodb_update_end_time
            }   
        })
        return data

    results = data_retrieval_by_time_mongodb(mongoclient)

    # Convert to Langchain format
    langchain_documents = [
        Document(
            page_content=doc["content"],
            metadata=doc.get("metadata", {})
        ) for doc in results
    ]

    # Splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(langchain_documents)
    # print(f"Total chunks created: {len(split_docs)}")
    # print("Example chunk:\n", split_docs[0].page_content)
    # print("Metadata:", split_docs[0].metadata)

    # Embedding
    hf = text_embedding_model_()
    texts = [doc.page_content for doc in split_docs]
    text_embeddings = hf.embed_documents(texts)
    return split_docs, text_embeddings


def weaviate_data_query():
    # Check if collection/class already exists, create one if not exists
    existing_collections = weaviate_client.collections.list_all()
    if weaviate_collection_name not in existing_collections:
        questions = weaviate_client.collections.create(
            name=weaviate_collection_name,
            properties=[
                wvc.config.Property(
                    name="content",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="source",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="timestamp",
                    data_type=wvc.config.DataType.DATE,
                )
            ]
        )
    
    split_docs, text_embeddings = text_split_embedding()

        # Store documents and vectors
    for doc, vector in zip(split_docs, text_embeddings):
        # 1. Check if exact content already exists
        existing = weaviate_client.collections.get("Supply_chain_material").query.fetch_objects(
            filters=Filter.by_property("content").equal(doc.page_content),
            return_properties=["content"]
        )

        if existing.objects:
            print("Exact content already exists. No action taken.")
        else:
            similar = weaviate_client.collections.get("Supply_chain_material").query.near_vector(
                near_vector=vector,
                certainty=weaviate_similarity_factor,
                return_properties=["content"]
            )

            if similar.objects:
                print("Similar vector exists. No action taken.")
            else:
                # Insert new object
                weaviate_client.collections.get("Supply_chain_material").data.insert(
                    properties={"content": doc.page_content},
                    vector=vector,
                    uuid = uuid4()
                )
                print("Inserted new object.")


def gemini_model(model_name, api_key):
    # Configure the API
    genai.configure(api_key = api_key)
    # Load the Gemini model
    gmodel = genai.GenerativeModel(model_name)
    return gmodel


def main():   
        
    model = gemini_model(gemini_model_name, google_api_key)
    # Ask a question
    query_question = 'Which sensors are currently used in prominent research and commercial vehicles?'
    hf = text_embedding_model_()
    query_vector = hf.embed_query(query_question)
    collection = weaviate_client.collections.get(weaviate_collection_name)
    results = collection.query.near_vector(query_vector, limit=10)
    retrieved_chunks = [obj.properties["content"] for obj in results.objects]

    # Gemini prompt
    context = "\n".join(retrieved_chunks)
    prompt = f"""Answer the question using the following context:

    Context:
    {context}

    Question:
    {query_question}

    Answer:"""
    response = model.generate_content(prompt)
    # Print the response
    print("Answer:")
    print(response.text)

    mongoclient.close()
    weaviate_client.close()


if __name__ == "__main__":
    main()