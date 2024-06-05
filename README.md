# local-llm
DSTA Internship Project

## Demo
Get the docker containers for the llm and neo4j database running and exposed to relevant ports.
Do `streamlit run streamlit-ui/jarvis.py` and open localhost:8501 on browser.

* You need to already have nlm-ingestor running in the background for point 1 to work. 
* You also need to have a docker container running ollama:llama3 and another container running the neo4j database.
1. Upload documents by storing them in the "/documents" folder in pdf-parsing directory and running document_parsing.ipynb
2. Navigate into streamlit-ui directory and run the command `streamlit run ui.py`.

Fine-tuning the prompt is extremely important.
