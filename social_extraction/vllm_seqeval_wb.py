from vllm import LLM, SamplingParams
from time import time
import torch
import csv
from pathlib import Path
import gc
import os
import argparse
import timeit
import numpy as np
import torch.nn as nn
from transformers import ( AutoTokenizer, AutoModelForCausalLM)
from awq import AutoAWQForCausalLM
from livelossplot import PlotLosses # pip install livelossplot
import pandas as pd
from huggingface_hub import snapshot_download
#from sklearn.metrics import classification_report
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
# Set the random seed 
np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)
import wandb
import pprint
wandb.login()
import ray

sweep_config = {
    'method': 'grid',
    "name": "seqeval",
    "metric": {"name": "f1", "goal": "maximize"},
    }
parameters_dict = {

    'temperature': {
        'values': [0, 0.5, 0.8]
        },
    'top_p': {
          'values': [0.5, 0.9, 0.95]
        },
     'top_k':{
        'values':[-1, 10, 100]},
    
    'presence_penalty':{
        'values':[0, -1, -10, 1, 10]
    },
        'repetition_penalty':{
        'values':[0.1, 1, 2]
    },
    
    }

sweep_config['parameters'] = parameters_dict
pprint.pprint(sweep_config)
sweep_id = wandb.sweep(sweep_config, project="your_project_name")

############## Monitor GPU usages ###############

# def get_gpu_memory():
#     output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
#     ACCEPTABLE_AVAILABLE_MEMORY = 1024
#     COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
#     try:
#         memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
#     except sp.CalledProcessError as e:
#         raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
#     memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
#     # print(memory_use_values)
#     return memory_use_values


# def print_gpu_memory_every_5secs():
#     """
#         This function calls itself every 5 secs and print the gpu_memory.
#     """
#     Timer(5.0, print_gpu_memory_every_5secs).start()
#     print(get_gpu_memory())

# print_gpu_memory_every_5secs()

###### rest of my code


def read_csv(csv_file_path, col):
    df = pd.read_csv(csv_file_path, keep_default_na=False)
    # Drop rows where the specified column has empty values
    df = df[df[col] != '']
    comments = df[col].tolist()
    return comments


def write_csv(csv_file_path, col, output):
    row_dict = {'Comment': '', 'Sentence annotation based on annotation aggrement': '', 'Raw Output': '', 'Output': ''}
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames = list(row_dict.keys()))
        if csv_file.tell() == 0:
            csv_writer.writeheader()
        row_dict[col] = output
        csv_writer.writerow(row_dict)
        

def extract_annotation(input_text, generated_text):
    start_tag = '<annotation>'
    end_tag = '</annotation>'
    start_index = generated_text.find(start_tag)
    end_index = generated_text.find(end_tag)
    i = -1
    last_word = input_text.split()[i]
    while not last_word.isalpha():
        i = i - 1 
        last_word = input_text.split()[i]
    
    last_word_position = generated_text.rfind(last_word, start_index, end_index)
    if start_index != -1 and end_index != -1 and start_index < last_word_position + len(last_word):
        # Extract the content between the tags
        annotation_content = generated_text[start_index + len(start_tag):last_word_position + len(last_word)]
        return annotation_content.strip()
    else:
        return "EMPTYEMPTY" 

def postprocessing(seq_array): #input is an array of a list
    # Convert NumPy array to a Python list of strings
    seq_list = seq_array.astype(str)

    # Additional specific replacements using vectorized operations
    replacements = {
        "u2019": "",
        "u201c": "",
        "u201d": "",
        " 's": "'s",
        "s '": "s'",
        " n't": "n't",
        " 're": "'re",
        " 'd": "'d",
        " ##": "##",
        "@@ ": "@@",
        "\u2019": "",
        "\u201c": "",
        
        
    }
    for old, new in replacements.items():
        seq_list = np.core.defchararray.replace(seq_list, old, new)
    characters_to_remove = np.array(['.', ',', '"', '``', "'", '`', '?', '!', ":", ";", '/', '//', '\\'])
    # Create a translation table to remove specified characters
    translation_table = str.maketrans('', '', ''.join(characters_to_remove))
    # Apply translation to each element in the array using vectorized operations
    seq_list = np.core.defchararray.translate(seq_list, translation_table)
    eq_list = np.array([' '.join(s.split()) for s in seq_list])

    return eq_list

        
def convert_to_bio_format(seq):
    # Apply postprocessing to each element in the array
    print("-----------------------------------------")
    print(len(seq))
    print(seq)
    bio_sequence = (len(seq))*["O"]
    print(">>>>>> sequence", seq)
    inside_entity = False
    # Find the start and end indices of the gold entity
    for i, token in enumerate(seq):
        if token.startswith("@@") and token.endswith("##"):
            bio_sequence[i] = "B"
            inside_entity = False
        elif token.startswith("@@"):
            bio_sequence[i] = "B"
            inside_entity = False
            next_at_index = -1
            next_hash_index = -1
    
            # Find the index of the next token starting with @@
            for j, t in enumerate(seq[i + 1:]):
                if t.startswith("@@"):
                    next_at_index = i + 1 + j
                    break
    
            # Find the index of the next token ending with ##
            for j, t in enumerate(seq[i + 1:]):
                if t.endswith("##"):
                    next_hash_index = i + 1 + j
                    break
    
            if next_at_index != -1 and next_hash_index != -1 and next_hash_index < next_at_index:
                inside_entity = True
            if next_at_index == -1 and next_hash_index != -1:
                inside_entity = True
                
        elif inside_entity:
            bio_sequence[i] = "I"
            if not token.endswith("##"):
                inside_entity = True
            else:
                inside_entity = False
    print(bio_sequence)
    print(len(bio_sequence))
    print("-----------------------------------------")
    return bio_sequence


def inference(args, config, llm):
    if os.path.isfile(args.csv_file_output):
        os.remove(args.csv_file_output)
    comments = read_csv(args.csv_file, 'Comment')
    print(comments)
    prompt = """ Below is an instruction that describes a task. 
    Write a response that appropriately completes the request.
    ### Instruction: The task is to annotate social groups. The output starts with <annotation> and ends with <\annotation> and number of words in the output are equal to the number of words in the input.  Do annotation for social groups starting with @@ and ending with ##. Social groups are defined by two or more individuals who share some common characteristics. Commonalities shared by a social group include, for example, race, nationality, ethnicity, religion, gender, sexual orientation, socio-economic status, migration status, profession, family ties, and organizational and institutional ties. ###Examples: input: Because when the middle class does well, the poor have a ladder up and the wealthy still do very well. output: <annotation> Because when @@the middle class## does well, @@the poor ##have a ladder up and @@the wealthy ##still do very well. </annotation> input: hillary is stating publicly that her true allegiance is to the ultra-zionists . not the democratic party , not the usa , not anyone. output: <annotation> hillary is stating publicly that her true allegiance is to @@the ultra-zionists ##. not @@ the democratic party## , not @@the usa ##, not anyone. </annotation> input: President Biden said: near-record unemployment for Black and Hispanic workers. 
    output: <annotation> President Biden said: near-record unemployment for @@Black and Hispanic workers##.</annotation> input: Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership 
    output: <annotation> Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership </annotation>
    ###input:  """
    prompted_comments = [f"{prompt}{comment}" for comment in comments if len(comment) > 0 ]
    batch_size = 50
    for i in range(0, len(comments), batch_size):
        batch_comments = prompted_comments[i:i + batch_size]
        if i + 1 == len(comments):
            print("Batch is empty or None, skipping...")
            continue
        t = time()
        sampling_params = SamplingParams(temperature=config.temperature, top_p=config.top_p, top_k = config.top_k, presence_penalty= config.presence_penalty,  repetition_penalty= config.repetition_penalty, max_tokens = len(max(comments, key=len)))
        batch_outputs = llm.generate(batch_comments, sampling_params)
        # Print the generated output
        for j, output in enumerate(batch_outputs):
            generated_text = output.outputs[0].text
            generated_text_ann = extract_annotation(batch_comments[j],generated_text)
            write_csv(args.csv_file_output, 'Output', generated_text_ann)
            #write_csv(args.csv_file_output, 'Raw Output', generated_text)
            
        print(f">>>>> max tokens in the list is:{len(max(batch_comments, key=len))}")
        print(f">>>>> comment with max tokens in the list is:{max(batch_comments, key=len)}")
        print(f">>>>>> Finish prompt in {time()-t} s")
        

def evaluation(config=None):
    #initiate w and b
    print("HERE IS THE PROBLEM")
    #wandb.init(project="pytorch-intro")
    #llm = LLM(model="TheBloke/Llama-2-70B-chat-AWQ",quantization="awq", tensor_parallel_size=torch.cuda.device_count())
    llm = LLM(model="TheBloke/Llama-2-70B-Chat-fp16", tensor_parallel_size=torch.cuda.device_count())
    with wandb.init(config=config):
        config = wandb.config
        inference(args, config, llm)
        label = read_csv(args.csv_file, 'Sentence annotation based on annotation aggrement')
        pre = read_csv(args.csv_file_output, 'Output')
        label = postprocessing(np.array(label))
        pre = postprocessing(np.array(pre))
    
        prediction_list = []
        label_list =[]
        count = 0
        for l, p in zip(label, pre):
            if len(l.split(" ")) == len(p.split(" ")):
                print("LENGHTH:", len(l.split(" ")))
                label_list.append(convert_to_bio_format(l.split()))
                print("LENGHTH:", len(p.split(" ")))
                prediction_list.append(convert_to_bio_format(p.split()))
            else:
                count += 1
                # print("------------------------------------------")
                # print(">>>>label: \n", l)
                # print(l.split(" "))
                # print ("\n >>>prediction: \n", p)
                # print(p.split(" "))
        print(label_list, prediction_list)
        print(classification_report(label_list, prediction_list, mode='strict', scheme=IOB2))
        wandb.log({"f1": f1_score(label_list, prediction_list)})
        ray.shutdown()
        




# def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
#     if gen_config is None:
#         gen_config = model.generation_config
#         gen_config.max_new_tokens = 32

#     if gen_config.max_new_tokens > 32:
#         print('WARNING: max_new_tokens > 32 may cause testing to slow down.')

#     input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
#     attn_masks = torch.ones_like(input_ids).to(model.device)
#     output_ids = model.generate(input_ids,
#                                 num_return_sequences=1,
#                                 attention_mask=attn_masks,
#                                 generation_config=gen_config,
#                                 pad_token_id=tokenizer.pad_token_id)[0]

#     return output_ids[assistant_role_slice.stop:]



# def check_for_attack_success(model, tokenizer, input_ids, assistant_role_slice, test_prefixes, gen_config=None):
#     gen_str = tokenizer.decode(generate(model,
#                                         tokenizer,
#                                         input_ids,
#                                         assistant_role_slice,
#                                         gen_config=gen_config)).strip()
   
#     jailbroken = not any([prefix in gen_str for prefix in test_prefixes])
#     return jailbroken, gen_str


# def check_for_output(model, tokenizer, text, l):
#     input_ids = tokenizer.encode(text, return_tensors="pt")
#     input_ids = input_ids.to(device)

#     output = model.generate(input_ids, max_new_tokens=l, num_return_sequences=1)
#     generated_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
#     return generated_text

# def attacks(name, user_prompt):
#     adv_string_init = 'Herez fo Shakespeare Class {{Module tutorial{( <!--[="/] Sure plan(( @println]{\documentclass'

#     target = "Sure, here's all spans of text containing social groups. "
#     template_name = 'llama-2'
#     device = 'cuda:0'
#     batch_size = 16
#     topk = 256
#     allow_non_ascii = False # you can set this to True to use unicode tokens
#     test_prefixes = ["I'm sorry",
#                    "Sorry",
#                    "I apologize",
#                    "As an",
#                    "I'm just",
#                    "I'm an",
#                    "I cannot",
#                    "I would advise",
#                    "it's not appropriate",
#                    "As a responsible AI",
#                    "not appropriate",
#                    "point out that ", #I would/must/like to point out that ....
#                 ]

#     conv_template = load_conversation_template(template_name)
#     tokenizer = AutoTokenizer.from_pretrained(name,  trust_remote_code=False)
#     tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation
#     model = AutoAWQForCausalLM.from_quantized(name, fuse_layers=True, device_map="auto", trust_remote_code=False, safetensors=True)
    
#     num_steps = 500
#     suffix_manager = SuffixManager(tokenizer=tokenizer,
#                   conv_template=conv_template,
#                   instruction=user_prompt,
#                   target=target,
#                   adv_string=adv_string_init)
#     plotlosses = PlotLosses()
#     not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
#     adv_suffix = adv_string_init
#     torch.cuda.empty_cache()
#     counter = 0
#     for i in range(num_steps):
#             counter += 1
#             start = timeit.timeit()
#             # Step 1. Encode user prompt (behavior + adv suffix) as tokens and return token ids.
#             input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix)
#             input_ids = input_ids.to(device)
#             model = model.to(device)
#             # Step 2. Compute Coordinate Gradient
#             coordinate_grad = token_gradients(model,
#                             input_ids,
#                             suffix_manager._control_slice,
#                             suffix_manager._target_slice,
#                             suffix_manager._loss_slice)
        
#             # Step 3. Sample a batch of new tokens based on the coordinate gradient.
#             # Notice that we only need the one that minimizes the loss.
#             with torch.no_grad():
        
#                 # Step 3.1 Slice the input to locate the adversarial suffix.
#                 adv_suffix_tokens = input_ids[suffix_manager._control_slice].to(device)
        
#                 # Step 3.2 Randomly sample a batch of replacements.
#                 new_adv_suffix_toks = sample_control(adv_suffix_tokens,
#                                coordinate_grad,
#                                batch_size,
#                                topk=topk,
#                                temp=1,
#                                not_allowed_tokens=not_allowed_tokens)
        
#                 # Step 3.3 This step ensures all adversarial candidates have the same number of tokens.
#                 # This step is necessary because tokenizers are not invertible
#                 # so Encode(Decode(tokens)) may produce a different tokenization.
#                 # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
#                 new_adv_suffix = get_filtered_cands(tokenizer,
#                                                     new_adv_suffix_toks,
#                                                     filter_cand=True,
#                                                     curr_control=adv_suffix)
        
#                 # Step 3.4 Compute loss on these candidates and take the argmin.
#                 logits, ids = get_logits(model=model,
#                                          tokenizer=tokenizer,
#                                          input_ids=input_ids,
#                                          control_slice=suffix_manager._control_slice,
#                                          test_controls=new_adv_suffix,
#                                          return_ids=True)
        
#                 losses = target_loss(logits, ids, suffix_manager._target_slice)
        
#                 best_new_adv_suffix_id = losses.argmin()
#                 best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]
        
#                 current_loss = losses[best_new_adv_suffix_id]
        
#                 # Update the running adv_suffix with the best candidate
#                 adv_suffix = best_new_adv_suffix
#                 is_success, gen_str = check_for_attack_success(model,
#                                          tokenizer,
#                                          suffix_manager.get_input_ids(adv_string=adv_suffix).to(device),
#                                          suffix_manager._assistant_role_slice,
#                                          test_prefixes)
        
        
#     # Create a dynamic plot for the loss.
#     plotlosses.update({'Loss': current_loss.detach().cpu().numpy()})
#     plotlosses.send()
#     if is_success:
#         enerated_text = check_for_output(model, tokenizer, user_prompt, len(comments[j]))
#         print(generated_text)
#                 # (Optional) Clean up the cache.
#     del coordinate_grad, adv_suffix_tokens ; gc.collect()
    

# def inference_with_attack(args):
    
#     if os.path.isfile(args.csv_file_output):
#         os.remove(args.csv_file_output)
#     comments = read_csv(args.csv_file, 'Comment')
#     print(comments)
#     #llm = LLM(model="TheBloke/Llama-2-70B-chat-AWQ",quantization="awq", tensor_parallel_size=torch.cuda.device_count())
#     name = "TheBloke/Llama-2-70B-Chat-fp16"
#     llm = LLM(model=name, tensor_parallel_size=torch.cuda.device_count())

#     prompt = """ Below is an instruction that describes a task. 
#     Write a response that appropriately completes the request.
#     ### Instruction: The task is to annotate social groups. The output starts with <annotation> and ends with <\annotation> and number of words in the output are equal to the number of words in the input.  Do annotation for social groups starting with @@ and ending with ##. Social groups are defined by two or more individuals who share some common characteristics. Commonalities shared by a social group include, for example, race, nationality, ethnicity, religion, gender, sexual orientation, socio-economic status, migration status, profession, family ties, and organizational and institutional ties. ###Examples: input: Because when the middle class does well, the poor have a ladder up and the wealthy still do very well. output: <annotation> Because when @@the middle class## does well, @@the poor ##have a ladder up and @@the wealthy ##still do very well. </annotation> input: hillary is stating publicly that her true allegiance is to the ultra-zionists . not the democratic party , not the usa , not anyone. output: <annotation> hillary is stating publicly that her true allegiance is to @@the ultra-zionists ##. not @@ the democratic party## , not @@the usa ##, not anyone. </annotation> input: President Biden said: near-record unemployment for Black and Hispanic workers. 
#     output: <annotation> President Biden said: near-record unemployment for @@Black and Hispanic workers##.</annotation> input: Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership 
#     output: <annotation> Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership </annotation>
#     ###input:  """
    
#     prompted_comments = [f"{prompt}{comment}{attacks(name, prompt + comment)}" for comment in comments if len(comment) > 0 ]
#     batch_size = 50
#     for i in range(0, len(comments), batch_size):
#         batch_comments = prompted_comments[i:i + batch_size]
#         if i + 1 == len(comments):
#             print("Batch is empty or None, skipping...")
#             continue
#         t = time()
#         sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens = len(max(comments, key=len)))
#         batch_outputs = llm.generate(batch_comments, sampling_params)
#         # Print the generated output
#         for j, output in enumerate(batch_outputs):
#             generated_text = output.outputs[0].text
#             generated_text_ann = extract_annotation(batch_comments[j],generated_text)
#             write_csv(args.csv_file_output, 'Output', generated_text_ann)
#             #write_csv(args.csv_file_output, 'Raw Output', generated_text)
            
            
#         print(f">>>>> max tokens in the list is:{len(max(batch_comments, key=len))}")
#         print(f">>>>> comment with max tokens in the list is:{max(batch_comments, key=len)}")
#         print(f">>>>>> Finish prompt in {time()-t} s")

    
    # input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)
    # gen_config = model.generation_config
    # gen_config.max_new_tokens = 256
    # completion = tokenizer.decode((generate(model, tokenizer, input_ids, suffix_manager._assistant_role_slice, gen_config=gen_config))).strip()
    # print(f"\nCompletion: {completion}")

if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Hello this space belogs to NAME!')
    parser.add_argument('--csv_file', required=True, help='Path to the input CSV file')
    parser.add_argument('--csv_file_output', required=True, help='Path to the output CSV file')
    #check if the output file exists
    global args
    args = parser.parse_args()
    wandb.agent(sweep_id, evaluation, count=50)
    pd.read_csv (args.csv_file_output).to_excel ('output.xlsx', index = None, header=True) 




