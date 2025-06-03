# llm_battery_material_supply_chain
## 1. Prerequisites
### 1.1 Weaviate
Weaviate is a hybrid search database with both semantic and keywords search included. It is open source and can be easily integrated with other packages such as MongoDB, HuggingFace (LLM), LangChain to support a variety of natural language processing (NLP)-related work. The recommended use of Weaviate is in Docker Container. Therefore, the docker needs to be installed as a first step. Atlair is the best user interface (UI) that is used to check if data is stored appropriately in Weaviate. The procedure is described as follows:


- install **Docker** and **Docker Compose**:
```
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker $USER
newgrp docker  # Or log out and log back in
```
- Run Weaviate with Docker:
i. Create a file named > docker-compose.yml:
```
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:1.24.10
    ports:
      - "8080:8080"
    restart: always
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-transformers'
      ENABLE_MODULES: 'text2vec-transformers'
      TRANSFORMERS_INFERENCE_API: 'http://t2v-transformers:8080'
      CLUSTER_ENABLED: 'false'
    volumes:
      - weaviate_data:/var/lib/weaviate

  t2v-transformers:
    image: semitechnologies/transformers-inference:sentence-transformers-paraphrase-multilingual-MiniLM-L12-v2
    environment:
      ENABLE_CUDA: '0'

volumes:
  weaviate_data:
```

ii. start Weaviate
Go to where Weaviate is installed and input the following:
```
sudo docker-compose up -d
```

iii. Check status and version

```
curl http://localhost:8080/v1/meta
```

- Set up Python environment 
This step ensures Weaviate can be connected in the Python environment.

```
python3 -m venv .venv
source .venv/bin/activate
pip install weaviate-client==3.24.1 langchain sentence-transformers
```
- Check if Weaviate connection is successful in Python environment 

```
import weaviate

client = weaviate.Client("http://localhost:8080")
if client.is_ready():
    print("✅ Weaviate connection is established.")
else:
    print("❌ Failed to connect to Weaviate.")

```

- Install and use Atlair for GraphQL (this is to verify if data can be successfully stored in Weaviate from external data sources)
The below provides the procedure to install and use Atlair as a Desktop app (Web app is also available at: [Link text](https://altair.sirmuel.design/)).
Please install Altair where Weaviate is installed using the following command:

```
sudo snap install altair   
```

After installing, opern Altair via Terminal by inputting ```Altair``` and hit ```Enter```. 

Note: after storing data to Weaviate, we can use the following steps to check if the data is stored successfully and correctly or not:

i. Set URL to: [Link text](http://localhost:8080/v1/graphql)
ii. run the following query to see the data stored in Weaviate. Here is just an example, the user can modify the bold part to query your own results.

<pre>
{
  Get {
    <b>ArticleChunk {
      content
    }</b>
  }
}
</pre>

iv. open weaviate and altair (altair is a user-friendly tool to check data query in Weaviate)
Input the code below in your terminal (where weaviate is installed) to connect Weaviate:
```
sudo docker-compose up
```
Execute the code below in your terminal (where weaviate is installed) to open Altair:
```
altair
```

### 1.2 Install packages
