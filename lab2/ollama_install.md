# Installing Ollama in /tmp/ollama (no sudo)

1. ~~Go to <https://github.com/ollama/ollama/releases/latest> and download `ollama-linux-amd64.tar.zst` (in Assets).~~ 
2. Copy the archive to /tmp/ollama and unpack it there:

```bash
mkdir -p /tmp/ollama
cp /opt/majewski/majewski/ollama/ollama-linux-amd64.tar.zst /tmp/ollama/
cd /tmp/ollama
tar x --zstd -f ollama-linux-amd64.tar.zst
```

3. Start the server, with the models stored in /tmp/ollama as well:

```bash
OLLAMA_MODELS=/tmp/ollama /tmp/ollama/bin/ollama serve
```

4. In a second terminal, download the models:

```bash
/tmp/ollama/bin/ollama pull qwen2.5:7b
/tmp/ollama/bin/ollama pull all-minilm
```

Keep the `ollama serve` terminal open while you work on the lab.
