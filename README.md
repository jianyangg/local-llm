# local-llm
DSTA Internship Project

## Demo (without using Docker Compose)

1. Get the docker containers for the llm and neo4j database running and exposed to relevant ports.
   - nlm-ingestor: `docker run -p 5001:5001 ghcr.io/nlmatics/nlm-ingestor:latest`
   - llm: Download Ollama locally or `docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 -e OLLAMA_NUM_PARALLEL=4 --name ollama ollama/ollama` as per this [tutorial](https://hub.docker.com/r/ollama/ollama).
   - neo4j: `docker run --publish=7474:7474 --publish=7687:7687 --volume=$HOME/neo4j/data:/data neo4j`

2. Run the following commands in separate terminals:
   - `streamlit run streamlit-ui/jarvis-ui/jarvis.py`
   - `python streamlit-ui/orchestrator/api.py`

   Then, open `localhost:8501` on your browser to access the Streamlit page.

Documentation is a work-in-progress.
