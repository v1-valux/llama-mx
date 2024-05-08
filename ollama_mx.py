#!/usr/bin/env python3

# pip install matrix-commander
# matrix-commander --login
# docker run -d -p 11434:11434 --gpus=all -v ollama:/root/.ollama --name ollama ollama/ollama
# docker run -d -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice

import requests
import asyncio
import json
import yaml
import re
import os
import sys
import base64
import nest_asyncio
import logging
from typing import List, Any
from subprocess import Popen, PIPE
from time import time, sleep
nest_asyncio.apply()

# Logging Config
logger = logging.getLogger()

class Config(object):
	'''
	Create config object.
	'''
	
	def __init__(self, filepath):
		'''
		Constructor.
		'''
		
		if not os.path.isfile(filepath):
			logger.error(f"Config file '{filepath}' does not exist")
		
		# Load in the config file at the given filepath
		with open(filepath) as file_stream:
			self.config = yaml.safe_load(file_stream.read())
		
		# Logging setup
		formatter = logging.Formatter(
			'%(asctime)s [%(levelname)s] %(message)s')
		
		log_level = self._get_cfg(["logging", "level"], default="INFO")
		logger.setLevel(log_level)
		
		file_logging_enabled = self._get_cfg(
			["logging", "file_logging", "enabled"], default=False)
		file_logging_filepath = self._get_cfg(
			["logging", "file_logging", "filepath"], default="bot.log")
		if file_logging_enabled:
			handler = logging.FileHandler(file_logging_filepath)
			handler.setFormatter(formatter)
			logger.addHandler(handler)
		
		console_logging_enabled = self._get_cfg(
			["logging", "console_logging", "enabled"], default=True)
		if console_logging_enabled:
			handler = logging.StreamHandler(sys.stdout)
			handler.setFormatter(formatter)
			logger.addHandler(handler)
		
		# Storage
		self.store_path = self._get_cfg(
			["storage", "store_path"], required=True)
		self.media_path = self._get_cfg(
			["storage", "media_path"], required=True)
		self.credentials_path = self._get_cfg(
			["storage", "credentials_path"], required=True)
		
		# Create media folder if it doesn't exist
		if not os.path.isdir(self.media_path):
			if not os.path.exists(self.media_path):
				os.mkdir(self.media_path)
			else:
				logger.error(f"storage.media_path '{self.media_path}' is not a directory")
		
		# Language
		self.language = self._get_cfg(
			["language"], default="en")
		
		# API
		self.stt_source = self._get_cfg(
			["stt", "source"], required=True, default='whisper-asr')
		self.stt_host = self._get_cfg(
			["stt", "host"], required=True, default='localhost')
		self.stt_port = self._get_cfg(
			["stt", "port"], default='9000')
		
		self.llm_source = self._get_cfg(
			["llm", "source"], required=True, default='ollama')
		self.llm_host = self._get_cfg(
			["llm", "host"], required=True, default='localhost')
		self.llm_port = self._get_cfg(
			["llm", "port"], default='11434')
		
		# Models
		self.llm_models = self._get_cfg(
			["llm", "models"], required=True, default=[])
		
		# Commands
		self.image_command = self._get_cfg(
			["commands", "visual_assist"], required=True, default='#cc')
		self.audio_command = self._get_cfg(
			["commands", "transcribe"], required=True, default='#cc')
		self.summary_command = self._get_cfg(
			["commands", "summarize"], required=True, default='#sum')
		self.help_command = self._get_cfg(
			["commands", "help"], required=True, default='#help')
		
		# Auto-Summarization
		self.auto_summary = self._get_cfg(
			["auto_summary"], required=True, default=False)
		self.auto_summary_words = self._get_cfg(
			["auto_summary_words"], required=True, default=200)

	def _get_cfg(
			self,
			path: List[str],
			default: Any = None,
			required: bool = True,
	) -> Any:
		'''
		Get a config option.
		'''
		# Sift through the the config until we reach our option
		config = self.config
		for name in path:
			config = config.get(name)
			# If at any point we don't get our expected option...
			if config is None:
				# Raise an error if it was required, allow default to be None
				if required:
					logger.error(
						f"Config option {'.'.join(path)} is required")
				# or return the default value
				return default
		
		# We found the option. Return it
		return config

##### I am lazy #####

# Read config file
if len(sys.argv) > 1:
	config_path = sys.argv[1]
else:
	config_path = 'config.yaml'

config = Config(config_path)

#####################

class MatrixBot():
	
	def __init__(self, **kwargs):
		'''
		Constructor.
		'''
		
		# necessary for credentials
		os.chdir(config.credentials_path)
		
		self.invoke_LLM = LLMPrompter()
		self.invoke_STT = STTPrompter()
		
		# save all available model prefixes as commands
		self.llm_commands = []
		for model in config.llm_models:
			self.llm_commands.append(model['prefix'])
		
		self.media_commands = [
			config.audio_command,
			config.image_command,
			config.summary_command
		]
		
		self.commands = ['#help']
		
		# append command list
		for command in self.media_commands:
			self.commands.append(command)
		for command in self.llm_commands:
			self.commands.append(command)
		
		self.sleep_duration = 1
	
	async def receive(self):
		'''
		Receiving events from the Matrix server.
		'''
		
		events, files, commands, texts, room_ids, sender_ids, related_events, related_files = [], [], [], [], [], [], [], []
		
		# get last messages
		
		#1 no download if file_urls of events get directly accessible over MC
		#process = Popen(['matrix-commander', '--no-ssl', '--store', config.store_path, '-l', 'ONCE', '-o', 'JSON'], stdout=PIPE, stderr=PIPE)
		
		# for now download and save all data for later use 
		process = Popen(['matrix-commander', '--no-ssl', '--store', config.store_path, '-l', 'ONCE', '--room-invites', 'JOIN', '--download-media', config.media_path, '--download-media-name', 'EVENTID', '-o', 'JSON'], stdout=PIPE, stderr=PIPE)
		stdout, stderr = process.communicate()
		
		# better to avoid program errors as input for this 'if' here?
		if stdout != b'':
			
			logger.debug(f'MC Output:{stdout.decode()}')
			
			# 
			try:
				
				event = json.loads(stdout.decode())
				
				# process room info for response
				event_id = event['source']['event_id']
				room_id = event['source']['room_id']
				room_name = event['room_display_name']
				sender_id = event['source']['sender']
				msg_body = event['source']['content']['body']
				msg_type = event['source']['content']['msgtype']
				
				# return room info
				room_ids.append(room_id)
				sender_ids.append(sender_id)
				
				# proceed only if output contains (somewhat valid) sender_id
				if sender_id.startswith('@'):
					
					logger.debug(f'Received text message: {msg_body}')
					
					
					if msg_type == 'm.text':
					
						audio_trigger = 'sent an audio file'
						image_trigger = 'sent an image'
						replied_to_audio = (audio_trigger in msg_body and any(x in msg_body for x in [config.audio_command,config.summary_command]))
						replied_to_image = (image_trigger in msg_body and config.image_command in msg_body)
						replied_to_media = (any(x in msg_body for x in self.media_commands))
						llm_triggered = (any(x in msg_body for x in self.llm_commands))
						
						# media commands
						if replied_to_audio or replied_to_image:
							
							# using the default matrix settings 
							# we receive a reply_to message as follows:
							# "> <@username:example.com> sent an audio file.\n\n{REPLY_MESSAGE}"
							# trying to filter out file_type and reply message
							
							trigger_sent = '> sent an'
							trigger_cr = '.\n\n'
							
							# get everything before and after ".\n\n"
							relation_msg = msg_body.split(trigger_cr)[0]
							reply_text = msg_body.split(trigger_cr)[-1]
							
							related_sender_id = relation_msg.replace('> ','').replace('<','').replace(audio_trigger,'').replace(image_trigger,'')
							related_event = event['source']['content']['m.relates_to']['m.in_reply_to']['event_id']
							
							## retrieve access token
							#with open(os.path.join(credentials_path,'credentials.json')) as file_stream:
							#	access_token = json.loads(file_stream.read())['access_token']
							
							# I planned to get media by the eventId via api call
							#process = Popen(['matrix-commander', '--no-ssl', '--access-token', access_token, '--store', config.store_path, "--rest", "get", "", f"https://{matrix_host}/_matrix/client/v3/rooms/{room_id}/events/{related_event}?access_token=access_token", '-o', 'JSON'], stdout=PIPE, stderr=PIPE)
							#stdout, stderr = process.communicate()
							# Load in the config file at the given filepath
							
							# get the first word after ".\n\n"
							command = reply_text.split(' ')[0]
							
							logger.debug(f'Triggered by reply to media with "{command}" command from {sender_id}.')
							
							# prompts should be min. 3 characters long
							if f'{command} ' in msg_body and len(reply_text.split(f'{command} ')[-1]) > 2:
								prompt = reply_text.split(f'{command} ')[-1]
								logger.debug(f'Custom text prompt provided: "{prompt}"')
							else:
								prompt = ''
							
							if replied_to_audio:
								
								related_filetype= 'audio'
								
								# logging only
								if command == config.summary_command:
									logger.debug(f'Processing summary of audio-file (sent by {related_sender_id}"): {related_event}')
								elif command == config.audio_command:
									logger.debug(f'Processing transcription of audio-file (sent by {related_sender_id}): {related_event}')
							
							elif replied_to_image:
								
								related_filetype= 'image'
								
								# loggin only
								logger.debug(f'Processing description of image-file (sent by {related_sender_id}): {related_event}')
							
							# pass related event
							related_events.append(related_event)
							# pass related filetype
							related_files.append(related_filetype)
							# pass command (mandatory)
							commands.append(command)
							# pass prompt (mandatory)
							texts.append(prompt)
						
						# LLM commands
						elif llm_triggered: 
							
							# get the first word after ".\n\n"
							command = msg_body.split(' ')[0]
							
							# get rid of prepending spaces in prompt
							if f'{command} ' in msg_body and len(msg_body.split(f'{command} ')[-1]) > 2:
								prompt = msg_body.split(f'{command} ')[-1]
								logger.debug(f'Triggered by LLM command "{command}" from {sender_id} with prompt: {prompt}')
								# pass command
								commands.append(command)
								# pass prompt
								texts.append(prompt)
							
							else:
								logger.debug(f'No prompt provided, skipping ..')
						
						elif msg_body.startswith(config.help_command):
							
							# set and pass help command
							command = config.help_command
							commands.append(command)
					
					elif msg_type == 'm.audio':
						if config.auto_summary == True:
							logger.debug('Triggered by audio file (auto-summarization enabled)')
							filetype='audio'
							
							# pass event_id and filetype
							events.append(event_id)
							files.append(filetype)
					
			except Exception as e:
				logger.debug(f"Parsing of multiple messages not yet implemented: {event}")
		
		return events, files, commands, texts, room_ids, sender_ids, related_events, related_files
	
	def retrieve_files(self, file_type, event_id):
		
		file_data = None
		
		#process = Popen(['matrix-commander', '--no-ssl', '--store', config.store_path, "--rest", "get", "", f"https://{matrix_host}/_matrix/client/v3/rooms/{room_id}/events/{rel_event}", '-o', 'JSON'], stdout=PIPE, stderr=PIPE)
		#stdout, stderr = process.communicate()
		
		file = os.path.join(config.media_path, event_id)
		
		# read audio stream
		if file_type == 'audio':
			logger.debug(f'Retrieving audio file: {file}')
			file_data = open(file, 'rb')
		
		# read image and convert to base64
		if file_type == 'image':
			logger.debug(f'Retrieving image file: {file}')
			with open(file, 'rb') as f:
				file_data = f.read()
			logger.info(f'Generating base64 string ..')
			file_data = base64.b64encode(file_data).decode("utf-8")
		
		return file_data
		
	def print_help(self):
		'''
		Help command /w model list
		'''
		
		_str = ''
		for i, r in enumerate(config.llm_models):
			if i == 0:
				_str += f"`{r['prefix']}`: {r['model_name']}"
			if i > 0:
				_str += '\n\n' + f"`{r['prefix']}`: {r['model_name']}"
		output = f'''
### Ollama-mx chat functions: \n\n
#### Voice Messages / Audio Files\n\n
Reply to an audio files with following options:\n\n
**{config.audio_command}** - transcribe voice messages to text\n\n
**{config.summary_command}** - summarize voice messages\n\n
#### Images\n\n
Reply to an image file with the following options:\n\n
**{config.image_command}** - describe whats on an image.\n\n
**{config.image_command}** **`prompt`** - ask a specific question about an image\n\n
#### Example\n
<mx-reply><blockquote><a href="">In reply to</a> <a href="">@user:example.com</a><br>sent an image.</blockquote></mx-reply>`{config.image_command} what could that meme have to do with Maths?`\n\n
\n\n\
#### LLM Prompts
Prompt a language model by using prefixes:\n\n
{_str}
\n\n
'''
		return output
		
	def model_name_available(data, model_name):
		'''
		Returns True if model is in list; not yet in use
		'''
		return next(d for d in data if d.get('model_name', None) == model_name)

	
	async def send(self, output, room_id):
		'''
		Sending response back to the Matrix server.
		'''
		# to avoid " conflicts
		output = output.replace('"', '\"')
		process = Popen(['matrix-commander', '--no-ssl', '--store', config.store_path, '--room', room_id, '--markdown', '-m', output], stdout=PIPE, stderr=PIPE)
		stdout, stderr = process.communicate()
	
	def start(self):
		
		logger.info(f'Bot started ..')
		logger.debug(f'Available commands: {self.commands}')
		
		while True:
			try:
				# receive events
				events, files, commands, texts, room_ids, sender_ids, related_events, related_files = asyncio.run(self.receive())
				output = None
				
				# any reply messages?
				if len(related_files) > 0:
					for current, event in enumerate(related_events):
						file_data = self.retrieve_files(file_type=related_files[current], event_id=event)
						if related_files[current] == 'audio':
							
							logger.debug(f'Received "{commands[current]}" command for audio file by "{sender_ids[current]}".')
							try:
								# call STT model
								stt_output = asyncio.run(self.invoke_STT.generate(data=file_data))
								output = stt_output
								
								if commands[current] == config.summary_command:
									
									# call LLM Model
									llm_output = asyncio.run(self.invoke_LLM.generate(event_type="text", data=commands[current], prompt=stt_output))
									output = llm_output
							
							except Exception as e:
								logger.exception(f"Exception occured: {e}")
							
							if output != None:
								logger.info(f'Sending response to {sender_ids[current]}')
								asyncio.run(self.send(output, room_ids[current]))
						
						elif related_files[current] == 'image':
							
							logger.debug(f'Received "{commands[current]}" command for image file by "{sender_ids[current]}".')
							try:
								output = asyncio.run(self.invoke_LLM.generate(event_type="image", data=file_data, prompt=texts[current]))
							except Exception as e:
								logger.exception(f"Exception occured: {e}")
							
							if output != None:
								logger.info(f'Sending response to {sender_ids[current]}')
								asyncio.run(self.send(output, room_ids[current]))
						else:
							logger.debug(f'Wrong media type')
				# any other commands used?
				elif len(commands) > 0:
					for current, command in enumerate(commands):
						
						# catch 'help' command
						if command == config.help_command:
							
							logger.debug(f'Received "{commands[current]}" command for audio file by "{sender_ids[current]}".')
							output = self.print_help()
						elif command in self.llm_commands and texts[current] != '':
							try:
								output = asyncio.run(self.invoke_LLM.generate(event_type="text", data=command, prompt=texts[current]))
							except Exception as e:
								logger.exception(f"Exception occured: {e}")
						
						if output != None:
							logger.info(f'Sending response to {sender_ids[current]}')
							asyncio.run(self.send(output, room_ids[current]))
				
				# or any files sent to process?
				elif len(files) > 0:
					for current, event in enumerate(events):
						if files[current] == 'audio':
							logger.debug(f'Received {files[current]} file by "{sender_ids[current]}". Checking word count ..')
							try:
								
								file_data = self.retrieve_files(file_type=files[current], event_id=event)
								# call STT model
								stt_output = asyncio.run(self.invoke_STT.generate(data=file_data))
								
								if len(stt_output.split(' ')) > config.auto_summary_words:
									logger.debug(f'Word count in audio file exceeds configured threshold ({len(stt_output.split(" "))}/{config.auto_summary_words}), summarizing "{sender_ids[current]}"\'s audio file ..')
									output = stt_output
									
									# call LLM Model
									llm_output = asyncio.run(self.invoke_LLM.generate(event_type="text", data=config.summary_command, prompt=stt_output))
									output = llm_output
								else:
									logger.debug(f'Word count in audio file is below configured threshold. ({len(stt_output.split(" "))}/{config.auto_summary_words}) ')
							except Exception as e:
								logger.exception(f"Exception occured: {e}")
							
							if output != None:
								logger.info(f'Sending response to {sender_ids[current]}')
								asyncio.run(self.send(output, room_ids[current]))
						
			except Exception  as e:
				logger.exception(f"Error in main loop: {e}")
				sys.exit(1)
			except KeyboardInterrupt:
				logger.exception("Received keyboard interrupt.")
				sys.exit(1)
			
			sleep(self.sleep_duration)

class LLMPrompter():
	
	def __init__(self):
		'''
		Constructor.
		
		'''
		
		# Set URL for the ollama server
		if config.llm_source == 'ollama':
			llm_url =  f'http://{config.llm_host}:{config.llm_port}/api/generate'
		
		# Set URL for the localai server
		if config.llm_source == 'localai':
			llm_url = f'http://{config.llm_host}:{config.llm_port}/v1/chat/completions'
		
		self.llm_api = llm_url
	
	async def generate(self, event_type, data, prompt):
		'''
		Generate response.
		'''
		
		model_name = None
		if event_type == 'text':
			
			# set model by command contained in data
			command = data
			for model in config.llm_models:
				# see if model command is in model prefix list
				if model['prefix'] in command:
					model_name = model['model_name']
			
			if command == config.summary_command:
				
				# see if a native model is available in model list
				native_models = []
				for model in config.llm_models:
					if model['language'] == config.language:
						native_models.append(model['model_name'])
				# take the first found native model
				model_name = native_models[0]
				
				# Multilingual Summary Agents
				if config.language == 'en':
					prompt = f'Voice message:\n\n"{prompt}"\n\n"Rephrase the message so that it contains the key messages as well as names, places or times and other relevant information briefly and concisely."'
				if config.language == 'de':
					prompt = f'Sprachnachricht:\n\n"{prompt}"\n\n"Formuliere die Nachricht so um, dass sie die Kernaussagen sowie Namen, Orte oder Uhrzeiten und andere relevante Informationen kurz und knapp enthält."'
				if config.language == 'it':
					prompt = f'Messaggio vocale:\n\n"{prompt}"\n\n"Riformula il messaggio in modo che contenga i messaggi chiave, i nomi, i luoghi o gli orari e altre informazioni rilevanti in modo breve e conciso."'
				if config.language == 'nl':
					prompt = f'Spraakbericht:\n\n"{prompt}"\n\n"Herformuleer het bericht zodat het kort en bondig de belangrijkste boodschappen en namen, plaatsen of tijden en andere relevante informatie bevat."'
				if config.language == 'fr':
					prompt = f'Message vocal:\n\n"{prompt}"\n\n"Reformule le message de manière à ce qu\'il contienne les messages clés ainsi que les noms, lieux ou heures et autres informations pertinentes de manière concise."'
				if config.language == 'es':
					prompt = f'Mensaje de voz:\n\n"{prompt}"\n\n"Reformula el mensaje para que contenga los mensajes clave, así como nombres, lugares u horas y otra información relevante de forma breve y concisa."'
				if config.language == 'gr':
					prompt = f'Φωνητικό μήνυμα:\n\n"{prompt}"\n\n"Επαναδιατυπώστε το μήνυμα έτσι ώστε να περιέχει τα βασικά μηνύματα καθώς και ονόματα, τόπους ή χρόνους και άλλες σχετικές πληροφορίες σύντομα και περιεκτικά."'
				if config.language == 'ru':
					prompt = f'Голосовое сообщение:\n\n"{prompt}"\n\n"Перефразируйте сообщение так, чтобы в нем коротко и ясно содержались ключевые слова, а также имена, места или время и другая необходимая информация."'
				if config.language == 'ua':
					prompt = f'Голосове повідомлення:\n\n"{prompt}"\n\n"Перефразуйте повідомлення так, щоб воно містило ключові ідеї, а також імена, місця, час та іншу важливу інформацію коротко і стисло."'
				
				# fallback model
				if model_name == None:
					model_name = config.llm_models[0]['model_name']
			
			# ollama api data
			if config.llm_source == 'ollama':
				
				prompt_data = {
					'model': model_name,
					'stream': False,
					'prompt': prompt,
					'keep_alive': '30m',
					'options': {
						'temperature': 0.2,
						'num_thread': 4,
						'num_gpu': 1,
						'main_gpu': 0,
						'low_vram': False
					}
				}
			
			# localai api data
			if config.llm_source == 'localai':
				
				prompt_data = {
					'model': model_name,
					'stream': False,
					'messages': [{'role': 'user', 'content': prompt}],
					'temperature': 0.3,
				}
  		
		if event_type == 'image':
			
			model_name = 'llava'
			
			base64_string = data
			
			# prompt for LLaVA
			if prompt == '':
				# LLaVA doesn't support multilingual inferencing.
				prompt = 'Describe in a few words what you see in the image.'
			
				#if config.language == 'de':
				#	prompt = 'Beschreibe in wenigen Worten, was du auf dem Bild siehst.'
			
			# pass the base64 string to llava
			# (ollama api data)
			prompt_data = {
				"model": model_name,
				"stream": False,
				"prompt": prompt,
				"images": [
					base64_string
				],
				'keep_alive': '10m',
				'options': {
					'num_thread': 4,
					'num_gpu': 1,
					'main_gpu': 0,
					'low_vram': False
				}
			}
		
		logger.info(f'Prompting "{model_name}" via {config.llm_source} API ..')
		logger.debug(f'Prompt: "{prompt}"')
		
		# Make a POST request to the server
		response_object = requests.post(self.llm_api, json = prompt_data)
		
		if response_object.status_code == 200:
			
			if config.llm_source == 'ollama':
				response = json.loads(response_object.text)['response']
			
			if config.llm_source == 'localai':
				response = json.loads(response_object.text)['choices'][0]['message']['content']
			
			logger.info(f'Response from ollama API: "{response}"')
			logger.debug(f'Response object: {response_object.text}')
			
		else:
			response = response_object.status_code
			logger.error(f'Bad response from ollama API: {response}')
			return None
		
		return response
		
class STTPrompter():
	
	def __init__(self):
		'''
		Constructor.
		
		'''
		
		# Set URL for stt server
		if config.stt_source == 'whisper-asr':
			stt_url = f'http://{config.stt_host}:{config.stt_port}/asr?encode=true&task=transcribe&language={config.language}&word_timestamps=false&output=txt'
		
		if config.stt_source == 'localai':
			stt_url = f'http://{config.stt_host}:{config.stt_port}/v1/audio/transcriptions'
		
		self.stt_api = stt_url
		
	async def generate(self, data):
		'''
		Generating response.
		'''
		
		# pass the audio-file to whisper
		file = {'audio_file': data}
		
		logger.info(f'Calling {config.stt_source} API with audio file ..')
		
		# Make a POST request to the server
		if config.stt_source == 'whisper-asr':
			response_object = requests.post(self.stt_api, files = file)
		
		if config.stt_source == 'localai':
			response_object = requests.post(self.stt_api, files = file, model = 'whisper-1')
		
		if response_object.status_code == 200:
			response = response_object.text
		
		else:
			response = ("Error: ", response_object.status_code)
		
		return response
