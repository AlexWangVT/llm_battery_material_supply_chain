###########################################################################################################
# This section used to make sure "streamlit" and "torch" have no conflict, must put at the top of this app
###########################################################################################################
import sys
import types

# Create a fake module path to prevent Streamlit from trying to inspect torch.classes
import torch
import time
# Monkey-patch torch.classes to trick Streamlit's watcher
if not hasattr(torch.classes, "__path__"):
    torch.classes.__path__ = []
    torch.classes.__file__ = "torch_classes_stub.py"
    torch.classes._is_streamlit_stub = True

# Optionally: remove torch.classes from sys.modules to fully avoid inspection
sys.modules["torch.classes"] = types.SimpleNamespace(__file__="torch_classes_stub.py", __path__=[])
##################################################################################################

import os
os.environ["USER_AGENT"] = "battery_materials_supply_chain_app" # set up USER_AGENT environment to avoid user_agent warning
import glob
import atexit
import yaml
import json
import nltk
import fitz  # PyMuPDF for better PDF parsing
import io
import re
import spacy   # 
import string
import datetime
import requests
import weaviate
import asyncio
import datetime
import accelerate
import unicodedata
from uuid import uuid4
from numpy import dot
from numpy.linalg import norm
import streamlit as st
from typing import List, Tuple
from bs4 import BeautifulSoup 
from datetime import timedelta
import weaviate.classes as wvc
from collections import Counter    #
from nltk.corpus import stopwords
import google.generativeai as genai
from weaviate.classes.query import Filter
from dotenv import load_dotenv
from pymongo import MongoClient
from datasets import Dataset
from langchain.schema import Document
from urllib.parse import urlparse, unquote, urljoin
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder 
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
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

############################################################################
load_dotenv(dotenv_path="../.env")
user_agent = os.getenv("USER_AGENT", "battery_materials_supply_chain_app")

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

############################################################################
# This section ensures mongodb and weaviate database close once at the end of the app run
def cleanup_clients():
    if "mongoclient" in st.session_state:
        st.session_state.mongoclient.close()
    if "weaviate_client" in st.session_state:
        st.session_state.weaviate_client.close()

atexit.register(cleanup_clients)
############################################################################
# Define environment
google_api_key = config["api"]["google_api"]
llama_api_key = config["api"]["llama_api"]
online_data_sources = config["database"]["online"]
local_pdf_database = config["database"]["local_pdf"]
mongoclient = MongoClient(config["database"]["mongodb_server"])
mongo_db = mongoclient[config["database"]["mongodb_database"]]
mycol = mongo_db[config["database"]["mongodb_collection_name"]]
weaviate_client = weaviate.connect_to_local() # connect to local temporarily, will connect to cloud server later
weaviate_collection_name = config["database"]["weaviate_collection_name"] # Collection name must have the first letter capitalized in Weaviate
weaviate_similarity_factor = config["database"]["weaviate_similarity_factor"]
text_embedding_model = config["model"]["text_embedding_model"]
text_embedding_model_kwargs = config["model"]["text_embedding_model_kwargs"]
gemini_model_name = config["model"]["gemini_model"]
llama_model_name = config["model"]["llama_model"]
cross_encoder_model = config["model"]["cross_encoder"]
english_stopwords = set(stopwords.words("english"))
chinese_stopwords = {"的", "了", "和", "是", "我", "不", "在", "有", "就", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要",
                    "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "它", "他", "她", "我们", "你们", "他们", "她们",
                    "它们", "与", "而", "但", "如果", "因为", "所以", "虽然", "但是", "或者", "即使", "那么", "这样", "那么", "而且",
                    "对", "从", "当", "在", "与", "比", "对比", "比起", "不如", "如", "如同", "同样", "一样", "更", "最", "多", "少",
                    "很", "太", "非常", "极", "极其", "相当", "几乎", "大约", "差不多", "差不多", "左右", "大概", "大约", "约", "将近",
                    "几", "些", "每", "所有", "全部", "一切", "任何", "某", "某些", "某个", "某些", "某种", "某些", "某类", "某种",
                    "其", "其余", "其余的", "其余的", "其他", "其他的", "其他人"}
UI_GARBAGE_PATTERNS = [
    r"(?i)\bCAPTCHA\b",
    r"(?i)(create account|have an account\? log in|sign up|log in|forgot your password\?|remember me)",
    r"(?i)(you're all set!|thank you.*?registered.*?logged in|check your email for details|invalid password or account does not exist)",
    r"(?i)(email address|password)",
    r"(?i)(ok|back|close|cancel|submit|next|previous|×)",  # common UI buttons and close icon
    r"(?i)(click here to unsubscribe|unsubscribe|subscribe now|subscribe to our newsletter|newsletter sign-up|newsletter)",
    r"(?i)(privacy policy|terms of service|cookie policy|cookie consent|terms and conditions|disclaimer)",
    r"(?i)(advertisement|ad closed by user|advertiser disclosure|advertise here|advertise)",
    r"(?i)(powered by|powered by wordpress|site map|contact us|about us|careers|contact information)",
    r"(?i)(loading\.{3,}|please wait|loading more posts|read more|continue reading|click here to read more)",
    r"(?i)(all rights reserved|copyright \d{4}|© \d{4}.*|©\d{4}.*|© \d{4}.*|©\d{4}.*)",
    r"(?i)(view more|related articles|related links|related posts|related products|you might also like)",
    r"(?i)(share on (facebook|twitter|linkedin|email)|share this article|follow us on|follow @\w+)",
    r"(?i)(error \d{3,4}|page \d+ of \d+|slide \d+ of \d+|posted on \w+ \d{1,2}, \d{4}|last updated on \w+ \d{1,2}, \d{4}|last modified on \w+ \d{1,2}, \d{4})",
    r"(?i)(this post was sponsored by|this article was written by|this entry was posted in|leave a comment|comments are closed)",
    r"(?i)(sidebar|footer|header|site navigation|homepage|navigation|skip to content)|^\s*confidential.*|^\s*(?:login|sign up|contact us)\s*$",
    r"(?i)(page \d+|\n\d+\n)|^\d{1,3}",  # standalone page numbers
    r"\.{3,}",  # multiple dots like ...
    r"[-_=]{3,}",  # multiple dashes/underscores
    r"\s{2,}",  # multiple spaces
]

pdf_chunk_max_token = 768
web_chunk_max_token = 512
chunk_overlapping = 100
web_search_max_depth = 2
CONVERSATION_HISTORY_FILE = '../data_sources/history.json'
conversation_history_turn_limit = -6
near_vector_limit = 100
rerank_top_k_chunks = 20
source_filter = None
# source_filter = ["../data_sources/local_pdfs/凯金——转让.pdf"] # This is used for testing purpose when we do not want to process all data, if all data need process, it equals None
# source_filter =["https://www.itechminerals.com.au/projects/campoona-graphite-project/", "https://www.itechminerals.com.au/projects/campoona-graphite-project/#elementor-action%3Aaction%3Dpopup%3Aclose%26settings%3DeyJkb19ub3Rfc2hvd19hZ2FpbiI6IiJ9"]


def crawl_nested_tabs(url, base_url, visited, web_search_max_depth, depth=0):
    
    if url in visited or depth > web_search_max_depth:
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
            results += crawl_nested_tabs(link, base_url, visited, web_search_max_depth, depth + 1)

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


def is_pdf_already_loaded(path, mycol):
        return mycol.count_documents({"source": path}) > 0


def extract_tables_from_pdf_text(text):
    """Extract tables using regex heuristics (or replace with Deep Learning model).
    This will improve text embedding."""
    lines = text.splitlines()
    tables = []
    current_table = []
    for line in lines:
        if re.search(r"\s{2,}", line):
            current_table.append(line)
        elif current_table:
            tables.append("\n".join(current_table))
            current_table = []
    if current_table:
        tables.append("\n".join(current_table))
    return tables


def extract_sections_from_text(text):
    """Simple section labeling based on heading patterns. This will improve text embedding."""
    section_heading_patterns = [
                        r"(?i)^([ivxlcdm]+)[\.\)]\s+.+$",                      # Roman numerals: I. Introduction
                        r"^(\d+(\.\d+)*[\.\)]?)\s+.+$",                        # Numeric: 1., 1.1, 1.2.3
                        r"(?i)^(chapter|section)\s+\d+[\.:]?\s+.+$",           # Chapter/Section 1:
                        r"^[A-Z\s]{5,}$",                                      # ALL CAPS headings
                        r"^[a-zA-Z][\.\)]\s+.+$",                              # Alphabetic bullets: a), b.
                        r"^\d+\s*[-–—]\s+.+$",                                 # Number - Title
                        r"^[一二三四五六七八九十百千]+[、.．． ]+.+$",             # Chinese numbered: 一、二、三
                        r"^[一二三四五六七八九十百千]+\.\d+[\s\.．]*.+$",         # Chinese numbered + sub: 一.1 方法
                        r"^(引言|背景|方法|研究现状|文献综述|讨论|结论|展望|附录|致谢)$",  # Common Chinese headings
                        r"^[（\(][一二三四五六七八九十百千]+[）\)]\s*.+$",         # Chinese bracket bullets: （一）
                        r"^(\d+\.){1,}\d*\s+.+$",                              # 1.1.1 Subsections
                        r"^\d+[:：]\s*.+$",                                    # 1：章节
                    ]

    def is_section_heading(text_line):
        for pattern in section_heading_patterns:
            if re.match(pattern, text_line.strip()):
                return True
        return False

    sections = []
    current = {"title": "Introduction", "content": ""}
    for line in text.splitlines():
        if is_section_heading(line):
            if current["content"]:
                sections.append(current)
            current = {"title": line.strip(), "content": ""}
        else:
            current["content"] += line + "\n"
    if current["content"]:
        sections.append(current)
    return sections


def mongodb(url, doc, metadata=None):
    def generate_title_from_url(url: str) -> str:
        path = urlparse(url).path
        title = path.strip("/").split("/")[-1].replace("_", " ")
        return unquote(title).title()

    def generate_title_from_filename(url: str) -> str:
        name = os.path.basename(url).replace("_", " ").replace("-", " ")
        return os.path.splitext(name)[0].title()

    if check_pdf_extension(url):
        sections = extract_sections_from_text(doc)
        tables = extract_tables_from_pdf_text(doc)
        title = generate_title_from_filename(url)
        pdf_document = {
                "content": doc,
                "metadata": {
                    **(metadata or {}),
                    "doc_type": "pdf",
                    "source": url,
                    "title": title,
                    "timestamp": datetime.datetime.now(datetime.UTC),
                    "sections": sections,
                    "tables": tables
                }
            }
        # insert pdf into mongodb
        # If existing_doc condition is another check about if the pdf is already loaded or not by checking the content of the pdf document. 
        # This second-level check aims to avoid the issue resulted from the is_pdf_already_loaded function when the source has some special symbol like "\n"
        existing_doc = mycol.find_one({"metadata.source": pdf_document["metadata"]["source"]})
        if existing_doc:
            if existing_doc.get("content", "") != pdf_document['content']:
                mycol.replace_one({"_id": existing_doc["_id"]}, pdf_document)
                print(f"{url} documents updated!")
            else:
                print(f"{url} document exists!")
        else:
            mycol.insert_one(pdf_document)
    
    else:
        # insert web articles into mongodb
        try:
            soup = BeautifulSoup(doc.page_content, "html.parser")
            html_text = soup.get_text("\n")
            sections = extract_sections_from_text(html_text)
        except Exception:
            sections = []

        title = generate_title_from_url(url)
        insert_docu = {
            "content": doc.page_content,
            "metadata": {
                "source": url,
                "title": title,
                "doc_type": "webpage",
                "timestamp": datetime.datetime.now(datetime.UTC),
                "sections": sections
            }
        }
        existing_doc = mycol.find_one({"metadata.source": insert_docu["metadata"]["source"]})
        if existing_doc:
            if existing_doc.get("content", "") != insert_docu['content']:
                mycol.replace_one({"_id": existing_doc["_id"]}, insert_docu)
                print(f"{url} documents updated!")
            else:
                print(f"{url} document exists!")
        else:
            mycol.insert_one(insert_docu)


def read_pdf_with_fitz(source):
    """
    Read local or online PDF using PyMuPDF (fitz).
    Returns full text and basic metadata.
    """
    def is_url(path):
        return urlparse(path).scheme in ("http", "https")

    def sanitize_text(text):
        # Normalize and strip surrogates
        if isinstance(text, str):
            # Encode with surrogatepass to catch lone surrogates
            return text.encode('utf-8', 'surrogatepass').decode('utf-8', 'ignore')
        return text
    
    try:
        source = sanitize_text(source)
    except Exception as e:
        print(f"Failed to sanitize source path: {e}")
        raise

    # Load PDF from local or URL
    try:
        if is_url(source):
            response = requests.get(source)
            pdf_bytes = io.BytesIO(response.content)
            doc = fitz.open("pdf", pdf_bytes)
        else:
            doc = fitz.open(source)
    except Exception as e:
        print(f"Failed to open PDF: {repr(source)} | {e}")
        raise

    # Extract full text
    full_text = ""
    for page in doc:
        text = page.get_text()
        text = sanitize_text(text)
        full_text += text

    # Extract simple metadata
    metadata = doc.metadata or {}
    metadata_clean = {k: sanitize_text(v) for k, v in metadata.items()}
    return full_text, metadata_clean


def search_load_data(online_data_sources, local_pdf_database):
    with open(online_data_sources, "r") as file:
        web_sources = [line.strip() for line in file.readlines() if line.strip()]

    pdf_sources = glob.glob(os.path.join(local_pdf_database, "*.pdf"))

    if source_filter == None:
        lines = web_sources + pdf_sources    
    else:
        # The line below is used to load those in case not successfully loaded at the beginning
        lines = source_filter

    for line in lines:
        #########################
            # Add script here to include company name for each "line", all sub-tabs under this "line" assumed to have the same company name

        #########################     
        if not check_pdf_extension(line): # Load web articles
            visited = set()
            all_tabs = crawl_nested_tabs(line, line, visited, web_search_max_depth)
            for tab in all_tabs:
                try: 
                    loader = WebBaseLoader(tab)
                    docs = loader.load()
                    doc = docs[0]
                    mongodb(tab, doc)
                except Exception as e:
                    print(f"Error loading {tab}: {e}")
            st.write(f"Load documents from web {line} and store in MongoDB successfully!")
        else:
            if is_pdf_already_loaded(line, mycol):
                print(f"Skipped (already in DB): {line}")
                continue 
            
            try:
                doc, metadata = read_pdf_with_fitz(line)
                mongodb(line, doc, metadata)
            except Exception as e:
                print(f"Error loading {line}: {e}")
            st.write(f"Load PDF documents from {line} and store in MongoDB successfully!")


def text_embedding_model_():      
    model_name = text_embedding_model
    model_kwargs = text_embedding_model_kwargs
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return hf


def text_split_embedding(start_date, end_date, pdf_max_token, web_max_token, overlapping):

    def data_retrieval_by_time_mongodb(start_date, end_date):
        start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
        end_datetime = datetime.datetime.combine(end_date, datetime.time.max)
        data = mycol.find({"metadata.timestamp": {
                "$gte": start_datetime,
                "$lte": end_datetime
            }   
        })
        return data

    def clean_garbage_text(text: str) -> str:
        for pattern in UI_GARBAGE_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        # Remove excessive whitespace, repeated headers/footers (simple heuristic: lines repeated multiple times)
        lines = text.split('\n')
        line_counts = {}
        for line in lines:
            stripped = line.strip()
            if stripped:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1
        repeated_lines = {line for line, count in line_counts.items() if count > 3}
        cleaned_lines = [line for line in lines if line.strip() not in repeated_lines]
        text = '\n'.join(cleaned_lines)
        # Remove page number patterns
        text = re.sub(r'page \d+(\s*of\s*\d+)?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\d+\n', '\n', text)  # standalone page numbers
        
        # Remove line separators and artifacts
        text = re.sub(r'[-_=]{3,}', '', text)
        text = re.sub(r'\.{3,}', '', text)
        
        # Remove multiple blank lines and excessive spaces
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\n+', '\n', text)

        # Fix broken lines ending with incomplete words (optional heuristic)
        text = re.sub(r"([a-z])\n([a-z])", r"\1 \2", text, flags=re.IGNORECASE)

        # Remove centered headings alone on a line (title in the middle)
        text = re.sub(r'\n\s{3,}[A-Z][^\n]+\n', '\n', text)

        # Fix lone single-letter artifacts or broken suffixes (e.g., "s\n")
        text = re.sub(r"(\w+)'s\s*\n", r"\1's ", text)

        return text.strip()

    def sentence_split(text: str) -> List[str]:
        pattern = r'(?<=[。！？!?；;])|(?<=\.\s)|(?<=\?\s)|(?<=!\s)'  # basic punctuation-based split
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    # Smart token-aware chunking
    def token_chunk(text: str, max_tokens: int = 512, overlap: int = 100) -> List[str]:
        sentences = sentence_split(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            token_len = len(tokenizer.tokenize(sentence))
            if token_len > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))  # Save current chunk first
                chunks.append(sentence.strip())  # Save long sentence as its own chunk
                current_chunk = []
                current_length = 0
                continue

            if current_length + token_len > max_tokens:
                chunks.append(" ".join(current_chunk))
                # Start new chunk with overlap
                if overlap > 0:
                    overlap_sentences = []
                    overlap_tokens = 0
                    for s in reversed(current_chunk):
                        t = len(tokenizer.tokenize(s))
                        if overlap_tokens + t > overlap:
                            break
                        overlap_sentences.insert(0, s)
                        overlap_tokens += t
                    current_chunk = overlap_sentences
                    current_length = overlap_tokens
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += token_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


    def is_duplicate(chunk: str, seen_hashes: set) -> bool:
        # Simple hash-based deduplication
        h = hash(chunk.strip().lower())
        if h in seen_hashes:
            return True
        seen_hashes.add(h)
        return False


    results = data_retrieval_by_time_mongodb(start_date, end_date)
    tokenizer = AutoTokenizer.from_pretrained(text_embedding_model)
    
    # ✅ Filter by metadata.source if source_filter is specified, filter is used just for testing
    filtered_results = []
    for doc in results:
        source = doc.get("metadata", {}).get("source", "")
        if source_filter is None or source in source_filter:
            filtered_results.append(doc)

    if not filtered_results:
        st.warning(f"No documents matched source_filter='{source_filter}' in the given date range.")
        return [], []
    
    ##############################################################################################
    # # Define two splitters for webpage and pdf, with different chunk size and overlap
    # webpage_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    # split_docs = []
    # for doc in filtered_results:
    #     raw_content = doc["content"]
    #     cleaned_content = clean_garbage_text(raw_content)

    #     # Detect doc type
    #     doc_type = doc.get("metadata", {}).get("doc_type", "webpage")  # default to webpage if not specified
    #     langchain_documents = Document(page_content=cleaned_content, metadata=doc.get("metadata", {}))

    #     if doc_type == "pdf":
    #         chunks = pdf_splitter.split_documents([langchain_documents])
    #     else:
    #         chunks = webpage_splitter.split_documents([langchain_documents])

    #     split_docs.extend(chunks)
##############################################################################################
    split_docs = []
    seen_hashes = set()
    for doc in filtered_results:
        raw_text = doc.get("content", "")
        cleaned = clean_garbage_text(raw_text)
        chunks = token_chunk(cleaned, max_tokens=pdf_max_token if doc.get("metadata", {}).get("doc_type") == "pdf" else web_max_token, overlap=overlapping)
        for chunk in chunks:
            if is_duplicate(chunk, seen_hashes):  # Filter duplicate chunks
                continue
            split_docs.append(Document(page_content=chunk, metadata=doc.get("metadata", {})))
    
    st.write(f"Text splitting is completed! Total chunks: {len(split_docs)}")

    # Embedding
    hf = text_embedding_model_()
    texts = [doc.page_content for doc in split_docs]
    text_embeddings = hf.embed_documents(texts)
    st.write(f"Text embedding is completed! Total chunks: {len(text_embeddings)}")
    return split_docs, text_embeddings


def weaviate_data_query(start_date, end_date, pdf_max_token, web_max_token, overlapping):
    # Check if collection/class already exists, create one if not exists
    # weaviate_client.collections.delete(weaviate_collection_name) # This is only used when you want to clean out and reset Weaviate database 
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
                ),
                wvc.config.Property( 
                    name="title",
                    data_type=wvc.config.DataType.TEXT,
                )
            ]
        )

    split_docs, text_embeddings = text_split_embedding(start_date, end_date, pdf_max_token, web_max_token, overlapping)

    # is_valid_query used to avoid stop words error
    def is_valid_query(text: str) -> bool:
        # Remove punctuation and lowercase the query
        print(f"🧪 Checking query: {text[:60]}...") 
        if not text or text.strip() == "":
            print("⚠️ Text is empty or whitespace only.")
            return False
        
        cleaned = text.translate(str.maketrans('', '', string.punctuation)).strip().lower()
        words = set(cleaned.split())
        chinese_chars = set(c for c in cleaned if c in chinese_stopwords)
        english_words = words - english_stopwords
        is_valid = bool(english_words or chinese_chars)
        print(f"✅ is_valid_query result: {is_valid}")
        return is_valid
    
    # Store documents and vectors in Weaviate
    for doc, vector in zip(split_docs, text_embeddings):
        if not is_valid_query(doc.page_content):
            print((f"⚠️ Skipped doc: Only stopwords or too short.\nContent: {doc.page_content[:200]}"))
            continue
        # 1. Check if exact content already exists
        try: 
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
                    metadata = doc.metadata or {}
                    properties = {
                        "content": doc.page_content,
                        "source": metadata.get("source", ""),
                        "title": metadata.get("title", ""),
                        "timestamp": metadata.get("timestamp", "")
                    }
                    weaviate_client.collections.get("Supply_chain_material").data.insert(
                        properties=properties,
                        vector=vector,
                        uuid = uuid4()
                    )
                    print(f"Successfully insert {metadata.get("source", "")} in Weaviate!")
        except Exception as e:
            print(f"❌ Exception occurred: {e}")
    st.write("Data successfully stored in Weaviate!")
    

def data_processing(start_date, end_date, pdf_max_token, web_max_token, overlapping):
    search_load_data(online_data_sources, local_pdf_database)
    st.write("Data search and load successfully completed!")
    weaviate_data_query(start_date, end_date, pdf_max_token, web_max_token, overlapping)
    mongoclient.close()
    st.write("Data update successfully completed!")


def llm(model_name, api_key):
    # Configure the API
    genai.configure(api_key = api_key)
    # Load the Gemini model
    gmodel = genai.GenerativeModel(model_name)
    return gmodel


def rewrite_query(query: str) -> str:
    # Prompt to rewrite vague or conversational queries
    REWRITE_PROMPT = PromptTemplate.from_template("""
                You are an assistant that rewrites user questions to improve semantic search performance in a technical RAG system.
                                                  
                Instructions:
                - If the original question is in Chinese, rewrite it in Chinese.
                - If the original question is in English, rewrite it in English.
                - Make the query more specific, use complete language, and include any implied terms relevant to battery materials and the supply chain domain.

                Original Question: "{query}"
                Rewritten Search-Friendly Question:
                """)
    llm = ChatGoogleGenerativeAI(model=gemini_model_name, google_api_key=google_api_key, temperature=0)
    rewrite_chain = REWRITE_PROMPT | llm

    return rewrite_chain.invoke({"query": query}).content.strip()


def question_and_answers(query_question, conversation_history): 
    
    def cosine_similarity(vec1, vec2):
        return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    def is_follow_up(prev_question: str, current_question: str, hf_embedding_model, threshold: float) -> bool:
        vec1 = hf_embedding_model.embed_query(prev_question)
        vec2 = hf_embedding_model.embed_query(current_question)
        similarity = cosine_similarity(vec1, vec2)
        return similarity >= threshold

    # --- VERIFICATION STEP ---
    def verify_answer(query_question, answer, context):
        verify_prompt = f"""
            You are a helpful assistant whose job is to verify the correctness of answers.

            Question:
            {query_question}

            Answer:
            {answer}

            Relevant Context:
            {context}

            Please do the following:

            1. Verify whether the answer is correct, complete, and supported by the context.
            2. Explain any issues, missing information, or inaccuracies if found.
            3. Suggest specific next questions or follow-up steps the user could ask to get more complete or clearer information.

            Provide your verification and suggestions clearly and concisely.

            Verification:"""

        verification_response = model.generate_content([{"role": "user", "parts": [verify_prompt]}])
        verification = verification_response.text.strip()
        return verification
    # --- END VERIFICATION ---

    model = llm(gemini_model_name, google_api_key)
    hf = text_embedding_model_()
    cross_encoder = CrossEncoder(cross_encoder_model, device='cpu')

    # Ask a question
    if conversation_history:
        last_question = conversation_history[-1][0]
        if is_follow_up(last_question, query_question, hf, threshold=0.6):  # 🔧 use combined only if follow-up
            combined_query = last_question + " " + query_question
        else:
            combined_query = query_question
    else:
        combined_query = query_question
    
    rewritten_query = rewrite_query(combined_query)
    # st.info(f"🔁 Rewritten Query: {rewritten_query}")
    query_vector = hf.embed_query(rewritten_query)
    collection = weaviate_client.collections.get(weaviate_collection_name)
    results = collection.query.hybrid(
                        query=rewritten_query,           # 🔴 Keyword query
                        vector=query_vector,            # 🔴 Semantic vector
                        alpha=.3,                      # 🔴 Hybrid balance: 0=keyword only, 1=vector only
                        limit=near_vector_limit,
                        return_properties=["content", "source", "timestamp", "title"]    
                    )
    
    # Retrieve and include metadata
    raw_chunks = []
    for obj in results.objects:
        content = obj.properties.get("content", "")
        source = obj.properties.get("source", "")
        title = obj.properties.get("title", "")
        formatted_metadata = f"Source: {source}\n\nTitle: {title}"
        raw_chunks.append({
            "content": content,
            "metadata": formatted_metadata,
            "full": f"Metadata:\n{formatted_metadata}\n\nContent:\n{content}"
        })

    chunk_pairs = [(rewritten_query, chunk["content"]) for chunk in raw_chunks]
    scores = cross_encoder.predict(chunk_pairs)
    reranked_chunks = sorted(zip(raw_chunks, scores), key=lambda x: x[1], reverse=True)
    top_chunks = [chunk["full"] for chunk, _ in reranked_chunks[:rerank_top_k_chunks]]  
    context = "\n\n".join(top_chunks)
    st.write(context)

    # Build previous turns into the prompt
    history_text = ""
    for user_q, assistant_a in conversation_history[conversation_history_turn_limit:]:  # limit to last 6 turns
        history_text += f"User: {user_q}\nAssistant: {assistant_a}\n"

# #####################################################################################
#     # Query with Refine strategy 
#     # First chunk
#     existing_answer = ""
#     for idx, chunk in enumerate(retrieved_chunks):
#         if idx == 0:
#             # Initial generation (include conversation history here)
#             prompt = f"""
        # You are a smart expert assistant helping the user answer questions. You have access to:

        # 1) Relevant Context: documents and metadata provided by the user.
        # 2) Your own general knowledge and facts up to your knowledge cutoff.

        # For every answer, combine both sources **even if the context is sufficient.**
        # When analyzing the relevant context, carefully examine **both the content and the attached metadata** for each chunk.  
        # Use the metadata to identify key entities such as company names, dates, locations, sources, titles, or other important details, and relate these to the information in the content.

        # Use the format below:
        # ---
        # **From Context**: [Clearly indicate what you found in the context, cite metadata if possible.]

        # **From General Knowledge**: [Add background, facts, or reasoning based on your own knowledge.]
        # ---

        # Do not skip either section unless the question is purely factual with no context overlap.

        # You may answer in English, Chinese, or both — choose whichever best conveys the information clearly and naturally, depending on the question and the context.
        
        # === Relevant Context ===
        # {context}

        # === Question ===
        # {query_question}

        # === Conversation History ===
        # {history_text}

        # Answer:"""
#             time.sleep(4)
#             response = model.generate_content([{"role": "user", "parts": [prompt]}])
#             existing_answer = response.text.strip()
#         else:
#             # Include conversation history only on the final chunk
#             if idx == len(retrieved_chunks) - 1:
#                 refine_prompt = f"""
#                             You are refining an answer based on new context and previous reasoning. Improve the previous answer if new context adds value.

#                             ---
#                             PREVIOUS ANSWER:
#                             {existing_answer}

#                             NEW CONTEXT:
#                             {chunk}

#                             CONVERSATION HISTORY:
#                             {history_text}

#                             QUESTION:
#                             {query_question}

#                             Only modify if new context provides additional information, corrections, or clarifications.
#                             ---

#                             Refined Answer:"""
#             else:
#                 refine_prompt = f"""
#                             You are refining an answer based on new context and previous reasoning. Improve the previous answer if new context adds value.

#                             ---
#                             PREVIOUS ANSWER:
#                             {existing_answer}

#                             NEW CONTEXT:
#                             {chunk}

#                             QUESTION:
#                             {query_question}

#                             Only modify if new context provides additional information, corrections, or clarifications.
#                             ---

#                             Refined Answer:"""
            
#             time.sleep(4)
#             response = model.generate_content([{"role": "user", "parts": [refine_prompt]}])
#             existing_answer = response.text.strip()
        # verification = verify_answer(query_question, existing_answer, context) # Verify after all chunks (more efficient)
#     return existing_answer, verification
#     #####################################################################################
    # Gemini prompt
    prompt = f"""
        You are a smart expert assistant helping the user answer questions. You have access to:

        1) Relevant Context: documents and metadata provided by the user.
        2) Your own general knowledge and facts up to your knowledge cutoff.

        For every answer, combine both sources **even if the context is sufficient.**
        When analyzing the relevant context, carefully examine **both the content and the attached metadata** for each chunk.  
        Use the metadata to identify key entities such as company names, dates, locations, sources, titles, or other important details, and relate these to the information in the content.

        Use the format below:
        ---
        **From Context**: [Clearly indicate what you found in the context, cite metadata if possible.]

        **From General Knowledge**: [Add background, facts, or reasoning based on your own knowledge.]
        ---

        Do not skip either section unless the question is purely factual with no context overlap.

        You may answer in English, Chinese, or both — choose whichever best conveys the information clearly and naturally, depending on the question and the context.
        
        === Relevant Context ===
        {context}

        === Question ===
        {query_question}

        === Conversation History ===
        {history_text}

        Answer:"""

    # Before each Gemini API call
    time.sleep(4)  # wait 4 seconds to keep under 15 requests/min
    response = model.generate_content([{"role": "user", "parts": [prompt]}])
    answer = response.text.strip() 
    verification = verify_answer(query_question, answer, context)

    return answer, verification


def llm_finetune(start_date, end_date):
    return st.write("Model finetuning successfully completed!")


def main():
    def save_history(history):
        history = history[-MAX_TURNS:]
        with open(CONVERSATION_HISTORY_FILE, "w") as f:
            json.dump(history, f)

    def load_history():
        if not os.path.exists(CONVERSATION_HISTORY_FILE):
            return []

        # Auto-delete if file is older than allowed
        file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(CONVERSATION_HISTORY_FILE))
        if datetime.datetime.now() - file_modified > timedelta(days=MAX_HISTORY_AGE_DAYS):
            os.remove(CONVERSATION_HISTORY_FILE)
            return []

        try:
            with open(CONVERSATION_HISTORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    
    MAX_HISTORY_AGE_DAYS = 7  # Auto-clear if file is older than this
    MAX_TURNS = 50  # Keep only recent 50 Q&A pairs
    st.set_page_config(page_title="Battery Materials Q&A", layout="centered")
    
    st.title("🔋 Battery Materials Supply Chain")

    # Sidebar or top dropdown for selecting mode
    mode = st.selectbox("Select Mode", ["💬 Q&A Mode", "💼 Data Processing Mode", "🛠️ Model Finetuning"])
    # Specify the date range for data updating
    default_start = datetime.datetime(2025, 5, 1)
    default_end = datetime.datetime(2025, 5, 17)
    if mode in ["💼 Data Processing Mode", "🛠️ Model Finetuning"]:
        st.markdown("### 📅 Select Data Time Range")
        date_range = st.date_input(
            "Select start and end date:",
            value=(default_start, default_end),
            min_value=datetime.date(2010, 1, 1),
            max_value=datetime.date.today() + datetime.timedelta(days=1)
        )
        # Only unpack when it's a tuple of two dates
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            # Use defaults until user selects both dates
            start_date, end_date = default_start, default_end

    if mode == "💼 Data Processing Mode":
        st.subheader("Run Data Ingestion and Embedding")

        if st.button("Run Data Pipeline"):
            with st.spinner("Processing data..."):
                try:
                    data_processing(start_date, end_date, pdf_chunk_max_token, web_chunk_max_token, chunk_overlapping)
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
    
    elif mode == "💬 Q&A Mode":
        st.subheader("Ask a Question")
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = load_history()

        with st.form(key="qa_form", clear_on_submit=False):
            query = st.text_input("Enter your question", key="query_input")
            submitted = st.form_submit_button("Get Answer")

        if submitted:
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Querying the model..."):
                    try:
                        start_time = time.time()
                        # ✨ Rewrite the query before retrieval
                        answers, verification = question_and_answers(query, st.session_state.conversation_history)
                        st.session_state.conversation_history.append((query, answers))  # <-- Maintain conversation
                        save_history(st.session_state.conversation_history)
                        
                        elapsed_time = time.time() - start_time
                        st.markdown(f"⏱️ Runtime: **{elapsed_time:.2f} seconds**")
                        st.markdown("### 🧠 Answer:")
                        st.write(answers or "Gemini returned no response.")
                        st.write(verification)

                        # Show chat history (optional)
                        with st.expander("🗂️ Conversation History"):
                            for user_q, assistant_a in st.session_state.conversation_history:
                                st.markdown(f"**User:** {user_q}")
                                st.markdown(f"**Assistant:** {assistant_a}")

                    except Exception as e:
                        st.error(f"⚠️ Error: {e}")

    elif mode == "🛠️ Model Finetuning":
        st.subheader("Model Finetuning")

        retrain = st.button("Re-train Embedding Model")
        if retrain:
            with st.spinner("Re-calibrating model..."):
                try:
                    llm_finetune(start_date, end_date)  # You need to define this function
                    st.success("✅ Model finetuning complete.")
                except Exception as e:
                    st.error(f"⚠️ Error during finetuning: {e}")

    weaviate_client.close()

if __name__ == "__main__":
    main()