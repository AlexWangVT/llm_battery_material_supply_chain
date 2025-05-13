import os
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
from weaviate.classes.query import Filter
from dotenv import load_dotenv
from pymongo import MongoClient
from datasets import Dataset
from langchain.schema import Document
from urllib.parse import urlparse, unquote, urljoin
from langchain.vectorstores import Weaviate
from weaviate.connect import ConnectionParams
from langchain.document_loaders import WebBaseLoader
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

load_dotenv()


def search_data(local_data_sources):
    with open(local_data_sources, "r") as file:
        lines = file.readlines()
        for line in lines:
            if ".pdf" not in line: # Load web articles
                loader = WebBaseLoader(line)
                documents = loader.load()
            else:
                loader = PyPDFLoader(line) # Load pdf files local or online
                async def load_pages():
                    pages = []
                    async for page in loader.alazy_load():
                        pages.append(page)
                    return pages
                pages = asyncio.run(load_pages())
                documents = "\n".join([p.page_content for p in pages])
                metadata = pages[0].metadata if pages else {}
    return documents


    
    
    print(f"{pages[0].metadata}\n")
    print(pages[0].page_content) 


def llm_model(model_name, token):
    # Use defined Langchain agent to search documents automatically
    # The agent is combined search tool with a LLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto",  # Automatically use GPU if available
                                                    torch_dtype=torch.float16, token=token)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256) 
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm



api_key = os.getenv("LLaMA_API_KEY")
model_name = "meta-llama/Llama-2-7b-chat-hf"
local_data_sources = "../data_sources/webpages.txt"
llm = llm_model(model_name, api_key)