# ollama-mx

A [matrix-commander](https://github.com/8go/matrix-commander) bot with AI functions for transcribing Voice Messages and prompting LLMs and LMMs (Large Multimodal Models) in [matrix](https://matrix.org/) chatrooms.

**Core functions:**
- transcribe audio files by replying to file with specific self-defined  (powered by [OpenAI-Whisper](https://openai.com/research/whisper))
- describe images by replying to image with self-defined commands and prompts (powered by [LLaVA LMM](https://llava-vl.github.io/))
- summarize audio-files / voice messagesby replying to file with self-defined commands.
- prompt a specific language model by using self-defined prefixes (like `#text` or `#code`)
- API support for LLMs/LMMs ([Ollama](https://ollama.com)), STT ([whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice)) and [LocalAI](https://localai.io/) for both.

All commands customizable.

**Known Bugs:**

- only media received while the bot is running can be processed  
- media is downloaded automatically and stored unencrypted


**Wish List:**

- get media files on demand (e.g. retrieving via rest-api)
- auto-delete media after some time
- chat-history support
- support more API sources

**Using the bot:**

```python
python start-ollama-mx.py /path/to/config.yaml

```

STT and LLM API services must be preinstalled.

The code might be ugly and things could be more safe and much cleaner but this is a hobby-project and my very first python program.

Framework by [#philipphoehn](https://github.com/philipphoehn).


If you like it, star it - If you don't, contribute or fork it. :)
