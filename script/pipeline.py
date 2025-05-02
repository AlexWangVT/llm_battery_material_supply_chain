import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
from pymongo import MongoClient
import weaviate
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from langchain.document_loaders import MongoDBLoader

# === Step 1: Ingest Raw Data ===
url = "https://example.com/article"
loader = WebBaseLoader(url)
documents = loader.load()

# === Step 2: Store Raw Data in MongoDB ===
mongo_client = MongoClient("mongodb://localhost:27017/")
raw_collection = mongo_client["domain"]["raw_documents"]
for doc in documents:
    raw_collection.insert_one({"content": doc.page_content, "metadata": doc.metadata})

# === Step 3: Retrieve Raw Data from MongoDB ===

loader = MongoDBLoader(
    connection_string="mongodb://localhost:27017/",
    db_name="domain",
    collection_name="raw_documents",
    content_field="content"
)
docs = loader.load()

# === Step 4: Chunk the Text ===
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overplap=50)
chunks = splitter.split_documents(docs)

# === Step 5: Generate Embeddings ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Step 6: Store Embeddings in Weaviate ===
weaviate_client = weaviate.Client("http://localhost:8080")
vectorstore = Weaviate(
    client=weaviate_client,
    index_name="DomainChunks",
    text_key="text",
    embedding=embedding_model
)
vectorstore.add_documents(chunks)

# === Step 7: Prepare Data for LLaMA2 Fine-Tuning ===
# Convert to instruction format for SFT
def format_example(chunk):
    return {
        "instruction": "Summarize the following:",
        "input": chunk.page_content,
        "output": chunk.page_content[:250] + "..."  # Dummy response
    }

formatted_data = [format_example(doc) for doc in chunks]
hf_dataset = Dataset.from_list(formatted_data)

# === Load Tokenizer and Model ===
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# === Apply LoRA for Efficient Training ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# === Tokenize ===
def tokenize(example):
    prompt = f"### Instruction:\n{example['instruction']}\n### Input:\n{example['input']}\n### Response:\n{example['output']}"
    tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = hf_dataset.map(tokenize)

# === Training Arguments and Trainer ===
training_args = TrainingArguments(
    output_dir="./llama2-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=300,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# === Fine-Tune the Model ===
trainer.train()
