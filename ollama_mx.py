#!/usr/bin/env python3

# pip install matrix-commander
# matrix-commander --login
# docker run -d -p 11434:11434 --gpus=all -v ollama:/root/.ollama --name ollama ollama/ollama
# docker run -d -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice

import requests
import asyncio
import json
import os
import sys
import base64
import nest_asyncio
import logging
from subprocess import Popen, PIPE
from time import time, sleep
nest_asyncio.apply()

# LOCAL SETTINGS
# stt api source
# choose between 'whisper-asr' (default), 'localai', and more (work in progress)
stt_source = 'whisper-asr'
stt_host = 'localhost'
# audio transcription language - change to '' for auto-detect
stt_language = 'en'

# llm api source
# choose between 'ollama' (default), 'localai', and more (work in progress)
llm_source = 'ollama'
llm_host = 'localhost'
llm_language = 'en'

llm_models = [{
	'prefix': '#eva',
	'model_name': 'eva-german',
	'language': 'de'
},
{
	'prefix': '#ml',
	'model_name': 'mistral-german',
	'language': 'de'
},
{
	'prefix': '#text',
	'model_name': 'llama3',
	'language': 'en'
},
{
	'prefix': '#code',
	'model_name': 'codellama',
	'language': 'en'
}]


image_command = '#cc'
audio_command = '#cc'
summary_command = '#sum'
help_command = '#help'

# Logging Config
logger = logging.getLogger(__name__)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

def log_debug(msg):
	logger.debug(msg)
	print('\033[90m' + 'DEBUG: ' + msg + '\033[0m')

def log_info(msg):
	logger.info(msg)
	print('\033[96m' + 'INFO: ' + msg + '\033[0m')

def log_warning(msg):
	logger.warning(msg)
	print('\033[93m' + 'WARNING: ' + msg + '\033[0m')

def log_error(msg):
	logger.error(msg)
	print('\033[91m' + 'ERROR: ' + msg + '\033[0m')

def log_exception(msg):
	logger.exception(msg)
	print('\033[91m' + 'ERROR: ' + msg + '\033[0m')

class MatrixBot():
	
	def __init__(self, **kwargs):
		'''
		Constructor.
		'''
		
		# folder where media and credentials are saved
		self.store_path = '/data/store'
		self.media_path = '/data/media'
		self.credentials_path = '/data'
		self.invoke_LLM = LLMPrompter()
		self.invoke_STT = STTPrompter()
		# necessary for credentials
		os.chdir(self.credentials_path)
		
		
		# save all available model prefixes as commands
		self.llm_commands = []
		for model in llm_models:
			self.llm_commands.append(model['prefix'])
		
		self.media_commands = [
			audio_command,
			image_command,
			summary_command
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
		
		commands, texts, room_ids, sender_ids, related_media_types, related_events = [], [], [], [], [], []
		
		# get last messages
		
		#1 no download if file_urls of events get directly accessible over MC
		#process = Popen(['matrix-commander', '--no-ssl', '--store', self.store_path, '-l', 'ONCE', '-o', 'JSON'], stdout=PIPE, stderr=PIPE)
		
		# for now download and save all data for later use
		process = Popen(['matrix-commander', '--no-ssl', '--store', self.store_path, '-l', 'ONCE', '--room-invites', 'JOIN', '--download-media', self.media_path, '--download-media-name', 'EVENTID', '-o', 'JSON'], stdout=PIPE, stderr=PIPE)
		stdout, stderr = process.communicate()
		
		# better to avoid program errors as input for this 'if' here?
		if stdout != b'':
			
			#log_debug(f'MC Output: \n\n{stdout.decode()}')
			
			# can we put this into a oneliner? `event = json.loads(tdout.decode()) `
			try:
				event = stdout.decode()
				event = json.loads(event)
			except Exception as e:
				log_exception(f"Exception occured while parsing message: {e}")
			
			# process room info for response
			room_id = event['source']['room_id']
			room_name = event['room_display_name']
			sender_id = event['source']['sender']
			msg_body = event['source']['content']['body']
			msg_type = event['source']['content']['msgtype']
			
			# return room info
			room_ids.append(room_id)
			sender_ids.append(sender_id)
			
			#log_debug(f{event})
			
			# proceed only if output contains (somewhat valid) sender_id
			if sender_id.startswith('@'):
				
				#log_debug(f'Received text message: {msg_body}')
				if msg_type == 'm.text':
					
					trigger_audio = 'sent an audio file.\n\n'
					trigger_image = 'sent an image.\n\n'
					
					triggers = [
						trigger_audio,
						trigger_image
					]
					
					
					# let's check if a message is of interest
					if any(x in msg_body for x in triggers) and any(x in msg_body for x in self.media_commands):
						
						log_info(f'Triggered by reply message from {sender_id}: {msg_body}')
						
						# using the default matrix settings 
						# we receive a reply_to message as follows:
						# "> <@username:example.com> sent an audio file.\n\n{REPLY_MESSAGE}"
						# trying to filter out media_type and reply message
						
						trigger_sent = '> sent an'
						trigger_cr = '.\n\n'
						
						# get everything before and after ".\n\n"
						relation_msg = msg_body.split(trigger_cr)[0]
						reply_text = msg_body.split(trigger_cr)[-1]
						
						# get the first word after ".\n\n"
						command = reply_text.split(' ')[0]
						prompt = reply_text.split(f'{command} ')[-1]
						
						related_event = event['source']['content']['m.relates_to']['m.in_reply_to']['event_id']
		
						# I planned to get media by eventId and an api call
						#process = Popen(['matrix-commander', '--no-ssl', '--store', self.store_path, "--rest", "get", "", f"https://{matrix_host}/_matrix/client/v3/rooms/{room_id}/events/{rel_event}", '-o', 'JSON'], stdout=PIPE, stderr=PIPE)
						#stdout, stderr = process.communicate()
						
						if trigger_audio in msg_body:
							
							related_media = 'audio'
							if command == summary_command:
								log_info(f'Summary of following audio-file requested by {sender_id}: {related_event}')
							elif command == audio_command:
								log_info(f'Processing of following audio-file requested by {sender_id}: {related_event}')
						
						elif trigger_image in msg_body:
								related_media = 'image'
								log_info(f'Processing of following image-file requested by {sender_id}: {related_event}')
								if len(prompt) > 0:
									log_debug(f'Added text prompt: {prompt}')

						related_events.append(related_event)
						related_media_types.append(related_media)
					
					elif any(x in msg_body for x in self.llm_commands):
						
						# get the first word after ".\n\n"
						command = msg_body.split(' ')[0]
						prompt = msg_body.split(f'{command} ')[-1]

						log_info(f'Triggered by text message from {sender_id}: {msg_body}')
						#log_debug(f'Command {command} recognized ..')
						
							
					elif '#help' in msg_body:
						
						# passing help command
						command = msg_body.split(' ')[0]
						prompt = msg_body.split(f'{command} ')[-1]
						
					# return data
					commands.append(command)
					texts.append(prompt)
						
		sleep(self.sleep_duration)
		
		return commands, texts, room_ids, sender_ids, related_media_types, related_events
	
	def retrieve_files(self, media_type, event_id):
		
		audio_data = None
		image_data = None
		
		#process = Popen(['matrix-commander', '--no-ssl', '--store', self.store_path, "--rest", "get", "", f"https://{matrix_host}/_matrix/client/v3/rooms/{room_id}/events/{rel_event}", '-o', 'JSON'], stdout=PIPE, stderr=PIPE)
		#stdout, stderr = process.communicate()
		
		file = os.path.join(self.media_path, event_id)
		
		if media_type == 'audio':
			#log_debug(f'Retrieving audio file: {file}')
			audio_data = open(file, 'rb')
		
		if media_type == 'image':
			#log_debug(f'Retrieving image file: {file}')
			with open(file, 'rb') as f:
				image_data = f.read()
			log_info(f'Generating base64 string ..')
			image_data = base64.b64encode(image_data).decode("utf-8")
		
		return audio_data, image_data
		
	def print_help(self):
		'''
		Help command /w model list
		'''
		
		_str = ''
		for i, r in enumerate(llm_models):
			if i == 0:
				_str += f"`{r['prefix']}`: {r['model_name']}"
			if i > 0:
				_str += '\n\n' + f"`{r['prefix']}`: {r['model_name']}"
		output = f'''
AI chat functions: \n\n\
- transcribe audio files by replying to file with `#cc`\n\
- summarize audio-files / voice messages by replying to file with `#sum`\n\
- describe images by replying to image with `#cc`\n\
- prompt a language model by using prefixes (no chat-history support yet)\n\
\n\
Available prefixes for llms: \n\n\
{_str}
'''
		return output
		
	def model_name_available(data, name):
		return next(d for d in data if d.get('name', None) == name)

	
	async def send(self, output, room_id):
		'''
		Sending response back to the Matrix server.
		'''
		# to avoid " conflicts
		output = output.replace('"', '\"')
		process = Popen(['matrix-commander', '--no-ssl', '--store', self.store_path, '--room', room_id, '--markdown', '-m', output], stdout=PIPE, stderr=PIPE)
		stdout, stderr = process.communicate()
	
	def start(self):
		
		log_info(f'Bot started ..')
		log_info(f'Available commands: {self.commands}')
		
		while True:
			# receive events
			commands, texts, room_ids, sender_ids, related_media_types, related_events = asyncio.run(self.receive())
			output = None

			if len(related_media_types) > 0:
				for current, event in enumerate(related_events):
					audio_data, image_data = self.retrieve_files(media_type=related_media_types[current], event_id=event)
					if related_media_types[current] == 'audio':
						try:
							# call STT model
							stt_output = asyncio.run(self.invoke_STT.generate(data=audio_data))
							output = stt_output
							
							if commands[current] == summary_command:
								# call LLM Model
								llm_output = asyncio.run(self.invoke_LLM.generate(event_type="text", data=commands[current], prompt=stt_output))
								output = llm_output
							
						except Exception as e:
							log_exception(f"Exception occured: {e}")
						
						if output != None:
							log_info(f'Sending response to {sender_ids[current]}')
							asyncio.run(self.send(output, room_ids[current]))

					elif related_media_types[current] == 'image':
						try:
							output = asyncio.run(self.invoke_LLM.generate(event_type="image", data=image_data, prompt=texts[current]))
						except Exception as e:
							log_exception(f"Exception occured: {e}")
						
						if output != None:
							log_info(f'Sending response to {sender_ids[current]}')
							asyncio.run(self.send(output, room_ids[current]))
					else:
						log_debug(f'fehler')
					
			elif len(texts) > 0:
				for current, text in enumerate(texts):
					
					# catch 'help' command
					if commands[current] == help_command:
						log_info(f'Received {help_command} message by {sender_ids[current]}')
						output = self.print_help()
					else:
						try:
							output = asyncio.run(self.invoke_LLM.generate(event_type="text", data=commands[current], prompt=text))
						except Exception as e:
							log_exception(f"Exception occured: {e}")
					
					if output != None:
						log_info(f'Sending response to {sender_ids[current]}')
						asyncio.run(self.send(output, room_ids[current]))
			
			#sleep(self.sleep_duration)

class LLMPrompter():
	
	def __init__(self):
		'''
		Constructor.
		
		'''
		
		# Set URL for the ollama server
		if llm_source == 'ollama':
			llm_url =  f'http://{llm_host}:11434/api/generate'
		
		if llm_source == 'localai':
			llm_url = f'http://{llm_host}:8080/v1/chat/completions'
		
		self.llm_source = llm_source
		self.llm_api = llm_url

	async def generate(self, event_type, data, prompt):
		'''
		Generating response.
		'''

		model_name = None
		if event_type == "text":
			
			# set model by command contained in data
			for model in llm_models:
				# see if model command is in model prefix list
				if model['prefix'] in data:
					model_name = model['model_name']
			
			if data == summary_command:
				#log_debug(f'Summary of voice message requested ..')
				
				for model in llm_models:
					# see if a multilingual model is available in model list
					if model['language'] == llm_language:
						model_name = model['model_name']

				# Multilingual Summary Prompts
				if llm_language == 'en':
					prompt = f'Give me a short summary of the following monologue: \n\n{prompt}'
				if llm_language == 'de':
					prompt = f'Gib mir eine kurze Zusammenfassung des folgenden Monologs: \n\n{prompt}'
				if llm_language == 'it':
					prompt = f'Riassumete brevemente il seguente monologo: \n\n{prompt}'
				if llm_language == 'nl':
					prompt = f'Geef me een korte samenvatting van de volgende monoloog: \n\n{prompt}'
				if llm_language == 'fr':
					prompt = f'Fais-moi un bref résumé du monologue suivant: \n\n{prompt}'

			if model_name == None:
				# fallback model (like above)
				model_name = llm_models[0]['model_name']
			
			if self.llm_source == 'ollama':
				
				prompt_data = {
					'model': model_name,
					'stream': False,
					'prompt': prompt,
					'keep_alive': '10m',
					'options': {
						'temperature': 0.3,
						'num_thread': 4,
						'num_gpu': 1,
						'main_gpu': 0,
						'low_vram': False
					}
				}
			
			if self.llm_source == 'localai':
				
				prompt_data = {
					'model': model_name,
					'stream': False,
					'messages': [{'role': 'user', 'content': prompt}],
					'temperature': 0.3,
				}
  		
		if event_type == 'image':

			model_name = 'llava'

			base64_string = data
			if l_language == 'de':
				prompt = 'Beschreibe in wenigen Worten, was du auf dem Bild siehst.'
			else:
				prompt = 'Describe in a few words what you see in the image.'

			# pass the base64 string to ollama
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

		log_info(f'Calling {model_name} via {llm_source} API with prompt: {prompt}')
		
		# Make a POST request to the server
		response_object = requests.post(self.llm_api, json = prompt_data)

		if response_object.status_code == 200:
			
			if self.llm_source == 'ollama':
				response = json.loads(response_object.text)["response"]
			
			if self.llm_source == 'localai':
				response = json.loads(response_object.text)['choices'][0]['message']['content']
			
			log_info(f'Response from ollama API: {response}')
			#log_debug(f'Response object: {response_object.prompt}')
			
		else:
			response = response_object.status_code
			log_error(f'Bad response from ollama API: {response}')
			return None
		
		return response
		
		
class STTPrompter():
	
	def __init__(self):
		'''
		Constructor.
		
		'''

		# Set URL for stt server
		if stt_source == 'whisper-asr':
			stt_url = f'http://{stt_host}:11435/asr?encode=true&task=transcribe&language={stt_language}&word_timestamps=false&output=txt'
		
		if stt_source == 'localai':
			stt_url = f'http://{stt_host}:8080/v1/audio/transcriptions'
		
		self.stt_source = stt_source
		self.stt_api = stt_url
		
	async def generate(self, data):
		'''
		Generating response.
		'''
		
		# pass the audio-file to whisper
		file = {'audio_file': data}
		
		log_info(f'prompting audio_file via {stt_source} api ..')
		
		# Make a POST request to the server
		if self.stt_source == 'whisper-asr':
			response_object = requests.post(self.stt_api, files = file)
		
		if self.stt_source == 'localai':
			response_object = requests.post(self.stt_api, files = file, model = 'whisper-1')
		
		if response_object.status_code == 200:
			response = response_object.text
		
		else:
			response = ("Error: ", response_object.status_code)
		
		return response
