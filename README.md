This is the *Building the Naive RAG* project I made myself.


<p>Movies whisk us away to magical realms and ignite our imagination. So why settle for just watching your favourite films when you can re-imagine them? In this project, you'll feed a large language model a dataset of movie scripts to craft brand-new scenes, character dialogues, and even invent entirely new twists.</p><br/><br/>Learn more at <a href="https://hyperskill.org/projects/518?utm_source=ide&utm_medium=ide&utm_campaign=ide&utm_content=project-card">https://hyperskill.org/projects/518</a>

Here's the link to the project: https://hyperskill.org/projects/518

Check out my profile: https://hyperskill.org/profile/554662699


Completed all five Hyperskill stages: Load, Split, Embed, Retrieve, and Generate.
The CLI downloads a selected movie script from IMSDb, normalizes whitespace,
splits at `INT.`, embeds scenes locally, retrieves five relevant scenes, and
uses a local chat model to answer or write a new scene.

- Python 3.14 was used for validation.
- FastEmbed 0.9.0 with `BAAI/bge-small-en-v1.5`: 384-dimensional embeddings.
- Qdrant 1.19.1: Cosine similarity and persistent, stable scene IDs.
- Qwen3 0.6B Q4_K_M: local rewriting and generation through LangChain.
- No paid API or credentials are required. Initial setup downloads model files.


Run commands from the repository root. Install Python and Docker separately.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

docker run -d --name naive-rag-qdrant \
  -p 127.0.0.1:6333:6333 \
  -v naive-rag-qdrant-storage:/qdrant/storage \
  -e QDRANT__TELEMETRY_DISABLED=true qdrant/qdrant:v1.19.1
```

If that project container already exists, use `docker start naive-rag-qdrant`
instead. Do not remove an existing database volume.

The startup script uses the standalone `llama-server` included in the official
Ollama macOS v0.40.7 bundle. It avoids the Ollama CLI's automatic identity-key
creation. Download the runtime and the public Qwen3 model separately:

```sh
mkdir -p .local/ollama .cache/chat-models
curl -fL https://github.com/ollama/ollama/releases/download/v0.40.7/ollama-darwin.tgz \
  -o /tmp/naive-rag-ollama-darwin.tgz
tar -xzf /tmp/naive-rag-ollama-darwin.tgz -C .local/ollama
curl -fL https://registry.ollama.ai/v2/library/qwen3/blobs/sha256:7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa \
  -o .cache/chat-models/qwen3-0.6b-q4_k_m.gguf
shasum -a 256 .cache/chat-models/qwen3-0.6b-q4_k_m.gguf
```

Verify the model digest is
`7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa`
before starting the server. Model information and license are available from
[Ollama's Qwen3 page](https://ollama.com/library/qwen3:0.6b).
The runtime expands to about 541 MiB; the model is about 523 MB.
FastEmbed also downloads its model (about 64 MiB) on first use.
These downloaded assets and database contents are excluded from Git.

In one terminal:

```sh
./start-local-chat.sh
```

In another terminal, with the virtual environment activated:

```sh
python "Building the Naive RAG/task/main.py"
```

Enter an exact movie title from the printed list, then a question or creative
request. The application prints the rewritten search, five retrieved scenes,
and `Final Answer:`. Stop the chat server with Ctrl-C. Both services bind only
to localhost: Qdrant on port 6333 and the chat model on port 11434.


All five stages were accepted through the Hyperskill PyCharm integration;
the final project panel showed 5 of 5 stages completed. The supplied Generate
test returned `(0, 'test OK')`. Course tests and metadata are managed by
Hyperskill and are not included in this source-only repository.

This is an educational prototype. Qwen3 0.6B can produce awkward rewrites,
repetition, or inaccurate prose. Passing the course checks does not establish
answer quality. The configured generation context is 32,768 tokens; very long
retrieved scenes can exceed it. Embedding models also have a limited input
length. Script availability and markup depend on the live IMSDb website.
Movie scripts remain the property of their respective rights holders and are
fetched at runtime, not redistributed in this repository.
