#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

echo "🔴 Retrieve LLAMA3 model..."
ollama pull llama3.1  # TODO: relies on the internet. Do this only for the initial setup.
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid