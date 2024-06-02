# ollama-mx

## Prototype - not for production use

A [matrix-commander](https://github.com/8go/matrix-commander) bot with AI functions for transcribing Voice Messages and prompting LLMs and LMMs (Large Multimodal Models) in [matrix](https://matrix.org/) chatrooms.

**Core functions:**
- transcribe audio files by replying to file with specific self-defined commands - needs [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice)
- ask about the contents of images by replying to an `m.image` message with a self-defined command (and prompt) - needs a [LLaVA](https://llava-vl.github.io/) or other LMM
- summarize audio-files / voice messages by replying to an `m.audio` message with self-defined commands.
- prompt a specific language model by using self-defined prefixes (like `#text` or `#code`)
- API support for LLMs/LMMs ([Ollama](https://ollama.com)), STT ([whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice)) or [LocalAI](https://localai.io/) for a combined solution.

All commands customizable via `config.yaml`.

**Known Quirks:**

- bot can will accept room-invitations automatically
- bot has access to all media files within joined rooms (and thus can theoretically leak data to other users via event_id bruteforcing)
- bot will only process the last of multiple messages received at the same time (e.g. startup after some downtime)
- only media files received while the bot is running can be processed
- media is downloaded automatically and stored unencrypted

**Feature Wish-List:**
- [x] auto summarize voice messages (depending on wordcount)
- [ ] automatically detect language for prompts
- [ ] get media files on demand (e.g. retrieving via rest-api)
- [ ] auto-delete media alternatively
- [ ] chat-history support

**Requirements:**

- docker
- matrix-commander

**Installation on Linux:**

```bash
pip install matrix-commander
matrix-commander --login

# fill out relevant information about your matrix-homeserver and choose a user impersonating your AI.

docker run -d -p 11434:11434 --gpus=all -v ollama:/root/.ollama --name ollama ollama/ollama
docker run -d -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice

```

**Using the bot:**

```python
python start-ollama-mx.py /path/to/config.yaml

```

The code might be ugly and things could be more safe and much cleaner but this is a hobby-project and my very first python program.

Framework by [#philipphoehn](https://github.com/philipphoehn).


If you like it, star it - If you don't, contribute or fork it. :)
