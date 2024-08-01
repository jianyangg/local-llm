# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install -r requirements.txt

RUN apt-get update

RUN apt-get install poppler-utils -y

EXPOSE 8501

# Install nltk
RUN pip3 install nltk
RUN [ "python", "-c", "import nltk; nltk.download('stopwords'); nltk.download('punkt')"]

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "jarvis.py", "--server.port=8501", "--server.address=0.0.0.0"]