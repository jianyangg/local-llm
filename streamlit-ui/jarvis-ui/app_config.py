# Toggle between docker and non-docker configurations
USE_DOCKER = False

if USE_DOCKER:
    config = {"ollama_base_url": "http://ollama:11434",
            "llm_name": "llama3:instruct",
            "nlm_url": "http://nlm-ingestor:5001/api/parseDocument?renderFormat=all&applyOcr=yes&useNewIndentParser=yes",
            "neo4j_url": "bolt://neo4j:7687",
            "neo4j_username": "neo4j",
            # This password is not the default password. Log in once and set the password to "password". Could be improved.
            "neo4j_password": "password",
            "orchestrator_url_entry": "http://orchestrator:5010/entry",
            "orchestrator_url_title": "http://orchestrator:5010/title",
            }
else:
    config = {"ollama_base_url": "http://localhost:11434",
            "llm_name": "llama3:instruct",
            "nlm_url": "http://localhost:5001/api/parseDocument?renderFormat=all&applyOcr=yes&useNewIndentParser=yes",
            "neo4j_url": "bolt://localhost:7687",
            "neo4j_username": "neo4j",
            # This password is not the default password. Log in once and set the password to "password". Could be improved.
            "neo4j_password": "password",
            "orchestrator_url_entry": "http://localhost:5010/entry",
            "orchestrator_url_title": "http://localhost:5010/title",
            }