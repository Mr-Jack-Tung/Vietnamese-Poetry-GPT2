# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Start: Wed 01 November 2023
# End: Sun 26 November 2023


import os, sys
import numpy as np
import torch
import random

from datetime import datetime
from tqdm import trange
from time import sleep

from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"]="False"

# https://arxiv.org/abs/2109.08203 - Torch.manual_seed(3407) is all you need
torch.manual_seed(3407)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the data
epoch = 2

# Total sample dataset ~ 1000 poems
poem_choice = 10

data_file_path = 'vietnamese_poems_1k.csv'
loss_margin = 3 # 4 # 3 # 2 # 1
learning_rate = 3e-4

# Step 1: Pretrained loading
model_name = 'Vietnamese-Poetry-GPT2-Model'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

# Step 2: Define the optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Step 3: Fine-tune the model
model.to(device)
model.train()

poem_data = []

df = pd.read_csv(data_file_path)
print("\n" + "File name: " + data_file_path)
print(df.info())
print("\n")

for i in range(len(df.index)):
	poem_data.append(df['poem'].loc[i])

print(f"Total poems: {len(poem_data)}")

poem_data_choice = random.sample(poem_data, poem_choice)
poem_data = poem_data_choice

print(f"Poems_choice: {len(poem_data)}")
print("\n")

for i in range(len(poem_data)):

	# print("\n" + poem_data[i]) # ~> to show the full content of the Poem
	print("\n" + poem_data[i].split("\n")[0]) # ~> to show the Title of the Poem

	try:
		input_ids = tokenizer.encode(text=poem_data[i], add_special_tokens=True, return_tensors='pt').to(device) 
	except Exception as e:
		print(f"tokenizer.encode Exception occurred", e)
		break

	description = "P{0}".format(len(poem_data)-i-1)

	# reset leaning rate
	optimizer.param_groups[0]['lr'] = learning_rate

	t = trange(epoch, desc=description, leave=True)
	for _ in t:
		t.refresh() # to show immediately the update
		sleep(0.1)

		try:
			loss = model(input_ids=input_ids, labels=input_ids)[0] # loss logits
		except Exception as e:
			print(f"model Exception occurred", e)
			break

		t.set_postfix(loss="{:.3f}".format(loss.item()))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()

		if loss < loss_margin:
			break

# Generate responses to new poem
model.eval()

print("\nSaving the model...")
tokenizer.save_pretrained(model_name)
model.save_pretrained(model_name)

def generate_poem(reques_poem):
	# Encode the reques_poem using the tokenizer
	input_ids = tokenizer.encode(reques_poem, add_special_tokens=False, return_tensors='pt').to(device)

	# Generate the response poem using the model
	sample_output = model.generate(input_ids, 
		pad_token_id=1, 
		eos_token_id=2, 
		# max_length=300, 
		max_new_tokens=350,
		do_sample=True, 
		repetition_penalty=1.2,
		).to(device)

	# Decode the generated response poem using the tokenizer
	answer = tokenizer.decode(sample_output[0], skip_special_tokens=True)

	return answer

# # Example usage

# prompt = "Ngày mai em đi"
# prompt = "Thì thầm mùa xuân"
# prompt = "Hoa cỏ may"
prompt = "Nắng chiều thu"
# prompt = "Xa em kỷ niệm"
reques_poem = "### Bài thơ: " + prompt + "\n"

response = generate_poem(reques_poem)

now = datetime.now() # current date and time
file_date_time = now.strftime("%Y%m%d-%H%M%S")

date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
print("\nTime:",date_time)
print(f"{response}")

