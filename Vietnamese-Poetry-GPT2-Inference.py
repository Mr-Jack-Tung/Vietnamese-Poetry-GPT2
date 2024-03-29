# -*- coding: utf-8 -*-
# Author: Mr.Jack _ Công ty www.BICweb.vn
# Start: Wed 01 November 2023
# End: Sun 26 November 2023


import torch
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer


# https://arxiv.org/abs/2109.08203 - Torch.manual_seed(3407) is all you need
torch.manual_seed(3407)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Step 1: Pretrained loading
model_name = 'Vietnamese-Poetry-GPT2-Model'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

# Generate responses to new poem 
model.eval()

def generate_poem(reques_poem):
	# Encode the reques_poem using the tokenizer
	input_ids = tokenizer.encode(reques_poem, add_special_tokens=False, return_tensors='pt').to(device)

	streamer = TextStreamer(tokenizer)

	# Generate the response poem using the model
	sample_output = model.generate(input_ids, 
		pad_token_id=1, 
		eos_token_id=2, 
		# max_length=300, 
		max_new_tokens=350,
		do_sample=True, 
		repetition_penalty=1.2,
		streamer=streamer,
		).to(device)

	# Decode the generated response poem using the tokenizer
	answer = ""
	answer = tokenizer.decode(sample_output[0], skip_special_tokens=True)

	return answer

prompt = ""
print("\n(Nhập 'quit!' để thoát chương trình)\n")

while prompt != "quit!":
	prompt = input("\nHuman:")

	if prompt != "quit!":
		now = datetime.now() # current date and time
		file_date_time = now.strftime("%Y%m%d-%H%M%S")
		date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
		
		print("\nTime:",date_time)
		print("AI:", end="")

		output = generate_poem(prompt)

