#!/bin/bash
SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
ollama_mx(){
    python3 $SCRIPT_DIR/start-ollama-mx.py
}
until ollama_mx; do
    echo "ollama-mx crashed with exit code $?. Restarting..." >&2
    sleep 1
done 
