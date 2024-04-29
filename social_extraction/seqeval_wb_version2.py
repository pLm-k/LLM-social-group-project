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
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
# Set the random seed 
np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)
import wandb
import pprint
wandb.login()
import ray
import re

sweep_config = {
    'method': 'grid',
    "name": "seqeval_wb_websci24",
    "metric": {"name": "f1", "goal": "maximize"},
    }
parameters_dict = {

    'temperature': {
        'values': [0, 0.5, 0.8, 1]
        },
    'top_p': {
          'values': [1]
        },
     'top_k':{
        'values':[-1]},
    
    'presence_penalty':{
        'values':[ 1]
    },
        'repetition_penalty':{
        'values':[1]
    },
        'prompt': {'values':[
    #         ###################### PROMPT_NER #################################
            """ ###Definition: Social groups are defined by two or more individuals who share some common characteristics. Commonalities shared by a social group include, for example, race, nationality, ethnicity, religion, gender, sexual orientation, socio-economic status, migration status, profession, family ties, and organizational and institutional ties.
    
    ###Example1:  Because when the middle class does well, the poor have a ladder up and the wealthy still do very well. 
    
    Answer: 
    1- the middle class|True| because its members share certain characteristics, experiences, and lifestyles that distinguish them from those in other income brackets
    2- the poor|True|as a social group is based on common economic conditions and shared experiences among its members. 
    3- the wealthy|True| The wealthy are considered a social group due to their shared economic status, lifestyle, and often, common interests
   
    output: <annotation> Because when @@the middle class## does well, @@the poor ##have a ladder up and @@the wealthy ##still do very well. </annotation> 

    
    Example2: hillary is stating publicly that her true allegiance is to the ultra-zionists . not the democratic party , not the usa , not anyone.  
    
    Answer: 
    1-hilary|False| "Hillary" typically refers to an individual, and it is not a social group. Hillary is a personal name and it is a person not a group and the size of "hilary" is one while the size of a group is bigger than one
    2-ultra-zionists|True| as it refers to  groups who hold extreme or radical views in support of Zionism
    3-democratic party|True| as the Democratic Party is a political party in the United States
    4-the usa|True| as it refers to the usa government not the geographical location of a country
    
    output: <annotation> hillary is stating publicly that her true allegiance is to the @@ultra-zionists## . not the @@democratic party## , not @@the usa##, not anyone. </annotation> 
    
    
    Example3: President Biden said: near-record unemployment for Black and Hispanic workers. 
    Answer: 
    1-President Biden|False| "President Biden" typically refers to an individual, and it is not a social group. President Biden is a personal name and it is a person not a group and the size of "President Biden" is one while the size of a group is bigger than one.
    2-Black and Hispanic workers|True| considered social groups based on shared racial or ethnic identities
    
    output: <annotation> President Biden said: near-record unemployment for @@Black and Hispanic workers##.
    
    Example4: Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership 
    Answer:
    1- Mr. Biden|False| as it refers to an individual, and it is not a social group. 
    2- Mr. Chaves|False|it is a person not a group and the size of "Mr. Chaves" is one while the size of a group is bigger than one.
    
    output: <annotation> Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership </annotation>
    ###input:  """, 
                             
                             ##################### PROMPT_NER_2 #################################
    """###Definition: Social groups are defined by two or more individuals who share some common characteristics. Commonalities shared by a social group include, for example, race, nationality, ethnicity, religion, gender, sexual orientation, socio-economic status, migration status, profession, family ties, and organizational and institutional ties.
    
    ###Example1:  Because when the middle class does well, the poor have a ladder up and the wealthy still do very well. 
    
    Answer: 
    First step:
    1- the middle class|True| because its members share certain characteristics, experiences, and lifestyles that distinguish them from those in other income brackets
    2- the poor|True|as a social group is based on common economic conditions and shared experiences among its members. 
    3- the wealthy|True| The wealthy are considered a social group due to their shared economic status, lifestyle, and often, common interests
    Second step: True social groups: the middle class, the poor, the wealthy
    Third step: output: <annotation> Because when @@the middle class## does well, @@the poor ##have a ladder up and @@the wealthy ##still do very well. </annotation> 

    Example2: hillary is stating publicly that her true allegiance is to the ultra-zionists . not the democratic party, not the usa , not anyone.  
    
    Answer: 
    First Step:
    1-hilary|False| "Hillary" typically refers to an individual, and it is not a social group. Hillary is a personal name and it is a person not a group and the size of "hilary" is one while the size of a group is bigger than one
    2-ultra-zionists|True| as it refers to  groups who hold extreme or radical views in support of Zionism
    3-democratic party|True| as the Democratic Party is a political party in the United States
    4-the usa|True| as it refers to the usa government not the geographical location of a country
   Second step: True social groups: the ultra-zionists, the democratic party, the usa
   Third step: output: <annotation> hillary is stating publicly that her true allegiance is to the @@ultra-zionists## . not the @@democratic party## , not @@the usa##, not anyone. </annotation> 
   
    
    
Example3: President Biden said: near-record unemployment for Black and Hispanic workers. 

    Answer: 
    First step:
    1-President Biden|False| "President Biden" typically refers to an individual, and it is not a social group. President Biden is a personal name and it is a person not a group and the size of "President Biden" is one while the size of a group is bigger than one.
    2-Black and Hispanic workers|True| considered social groups based on shared racial or ethnic identities
    Second step:True social groups:Black and Hispanic workers
    Third step: output: <annotation> President Biden said: near-record unemployment for @@Black and Hispanic workers##. </annotation>
    
Example4: Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership 
    Answer:
    First step:
    1- Mr. Biden|False| as it refers to an individual, and it is not a social group. 
    2- Mr. Chaves|False|it is a person not a group and the size of "Mr. Chaves" is one while the size of a group is bigger than one.
    Second step: True social groups: NONE
    Third step: output: <annotation> Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership </annotation>
    ###input:  """  , 
            ###################### COT #################################
            
            """Determine whether the text below contains mentions of social group. First, identify social groups based on this definition: Social groups are defined by two or more individuals who share some common characteristics. Commonalities shared by a social group include, for example, race, nationality, ethnicity, religion, gender, sexual orientation, socio-economic status, migration status, profession, family ties, and organizational and institutional ties. Second, do annotation for social groups starting with @@ and ending with ##.  The answer is in the followwing format:
The output starts with <annotation> and ends with <\annotation> and number of words in the output are equal to the number of words in the input.  

Below are some examples:

Text: Because when the middle class does well, the poor have a ladder up and the wealthy still do very well. 
Answer: <annotation> Because when @@the middle class## does well, @@the poor ##have a ladder up and @@the wealthy ##still do very well. </annotation>
Reason: the middle class, the poor, and the wealthy each one has a size of the group more than one and each group shares some characteristics.



Text: hillary is stating publicly that her true allegiance is to the ultra-zionists . not the democratic party , not the usa , not anyone. 
Answer: <annotation> hillary is stating publicly that her true allegiance is to the @@ultra-zionists## . not the @@democratic party## , not @@the usa##, not anyone. </annotation> 
Reason: since the ultra-zionists, the democratic party, and the USA each one has a size of the group more than one, and each group shares some characteristics.


Text: President Biden said: near-record unemployment for Black and Hispanic workers. 
Answer: <annotation> President Biden said: near-record unemployment for @@Black and Hispanic workers##.</annotation>
Reason: Black and Hispanic workers is defined as a social group since it has a size of the group more than one, and each group shares some characteristics.

Text: Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership 
Answer: <annotation> Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership </annotation>
Reason: There is no mention of social groups in the text

Text:  """, 

                    ###################### GPT_NER #################################
            
"""Below is an instruction that describes a task. 
    Write a response that appropriately completes the request.
    ### Instruction: The task is to annotate social groups. The output starts with <annotation> and ends with <\annotation> and number of words in the output are equal to the number of words in the input.  Do annotation for social groups starting with @@ and ending with ##. Social groups are defined by two or more individuals who share some common characteristics. Commonalities shared by a social group include, for example, race, nationality, ethnicity, religion, gender, sexual orientation, socio-economic status, migration status, profession, family ties, and organizational and institutional ties. ###Examples: input: Because when the middle class does well, the poor have a ladder up and the wealthy still do very well. output: <annotation> Because when @@the middle class## does well, @@the poor ##have a ladder up and @@the wealthy ##still do very well. </annotation> input: hillary is stating publicly that her true allegiance is to the ultra-zionists . not the democratic party , not the usa , not anyone. output: <annotation> hillary is stating publicly that her true allegiance is to the @@ultra-zionists## . not the @@democratic party## , not @@the usa##, not anyone. </annotation> input: President Biden said: near-record unemployment for Black and Hispanic workers. 
    output: <annotation> President Biden said: near-record unemployment for @@Black and Hispanic workers##.</annotation> input: Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership 
    output: <annotation> Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership </annotation>
    ###input:  """
#             ,   
#                              ######################  Fulcra #################################
# """ ###Objective: This is an annotation task to annotate social groups. Instructions are as follows.
# Step 1. Familiarization with the definition of social groups:  Social groups are defined by two or more individuals who share some common characteristics. Commonalities shared by a social group include, for example, race, nationality, ethnicity, religion, gender, sexual orientation, socio-economic status, migration status, profession, family ties, and organizational and institutional ties. Please make sure that you fully understand the definition of the social groups. 
# Step 2. Reading and Understanding the input: Read the entire input, ensuring you fully understand the context and grasp the intent behind the text, making notes. 
# Step 3. Social Groups Identification: For the input, identify social groups based on the given definition. If a social group has no apparent connection with the input, label it as ¨No connection¨do not include it in the final result list.
# Step 4. Social group Annotation: The output starts with <annotation> and ends with <\annotation> and number of words in the output are equal to the number of words in the input.  Do annotation for social groups starting with @@ and ending with ##.
    
# ###Examples: input: Because when the middle class does well, the poor have a ladder up and the wealthy still do very well. output: <annotation> Because when @@the middle class## does well, @@the poor ##have a ladder up and @@the wealthy ##still do very well. </annotation> input: hillary is stating publicly that her true allegiance is to the ultra-zionists . not the democratic party , not the usa , not anyone. output: <annotation> hillary is stating publicly that her true allegiance is to the @@ultra-zionists## . not the @@democratic party## , not @@the usa##, not anyone. </annotation> input: President Biden said: near-record unemployment for Black and Hispanic workers. output: <annotation> President Biden said: near-record unemployment for @@Black and Hispanic workers##.</annotation> input: Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership 
# output: <annotation> Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership </annotation>
#     ###input:  """ 
        
]} ,
    
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
    df = pd.read_csv(csv_file_path, keep_default_na=False, encoding='latin-1')
    # Iterate over each column
    for col in df.columns:
        # Clean the data by replacing 'ï¿½' with an empty string
        df[col] = df[col].str.replace('ï¿½', '')

    
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

    # start_tag = '<annotation>'
    # end_tag = '</annotation>'
    # start_index = generated_text.find(start_tag)
    # end_index = generated_text.find(end_tag)
    # i = -1
    # last_word = input_text.split()[i]
    # while not last_word.isalpha():
    #     i = i - 1 
    #     last_word = input_text.split()[i]
    
    # last_word_position = generated_text.rfind(last_word, start_index, end_index)
    # if start_index != -1 and end_index != -1 and start_index < last_word_position + len(last_word):
    #     # Extract the content between the tags
    #     annotation_content = generated_text[start_index + len(start_tag):last_word_position + len(last_word)]
    #     return annotation_content.strip()
    # else:
    #     return "EMPTYEMPTY" 


    
    input_text = input_text.lower()
    generated_text = generated_text.lower()
    patterns = [
        r'(?<=<annotation>)(.*?)(?=</annotation>)',
        r'(?<=<annotation>)(.*?)(?=\n)',
        r'(?<=<annotation>)(.*?)(?=<\/annotation>)'
    ]

    for pattern in patterns:
        match = re.search(pattern, generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
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
    comments = read_csv(args.csv_file, 'body_cleaned')
    print(comments)
    prompted_comments = [f"{config.prompt}{comment}" for comment in comments if len(comment) > 0 ]
    batch_size = 100
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
            print(generated_text)
            generated_text_ann = extract_annotation(batch_comments[j], generated_text)
            print("-----------")
            print(generated_text_ann)
            print("-----------")
            write_csv(args.csv_file_output, 'Output', generated_text_ann)
            #write_csv(args.csv_file_output, 'Raw Output', generated_text)
            
        #print(f">>>>> max tokens in the list is:{len(max(batch_comments, key=len))}")
        #print(f">>>>> comment with max tokens in the list is:{max(batch_comments, key=len)}")
        #print(f">>>>>> Finish prompt in {time()-t} s")
        

def evaluation(config=None):
    #initiate w and b
    print("HERE IS THE PROBLEM")
    #wandb.init(project="pytorch-intro")
    #llm = LLM(model="TheBloke/Llama-2-70B-chat-AWQ",quantization="awq", tensor_parallel_size=torch.cuda.device_count())
    llm = LLM(model="TheBloke/Llama-2-70B-Chat-fp16", tensor_parallel_size=torch.cuda.device_count())
    with wandb.init(config=config):
        config = wandb.config
        inference(args, config, llm)
        label = read_csv(args.csv_file, 'gold annotation')
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
                print("\n >>>>>>>label list:", label_list)
                print(">>>>>>>prediction list", prediction_list)
            else:
                count += 1
                print("------------------------------------------")
                print(">>>>label: \n", l)
                # print(l.split(" "))
                print ("\n >>>prediction: \n", p)
                # print(p.split(" "))
                print("\n >>>>>>>label list:", label_list)
                print(">>>>>>>prediction list", prediction_list)
        #print(classification_report(label_list, prediction_list, mode='strict', scheme=IOB2))
        #wandb.log({"accuracy": accuracy_score(label_list, prediction_list)})
        wandb.log({"f1": f1_score(label_list, prediction_list,  mode='strict', scheme=IOB2, average= 'micro')})
        wandb.log({"recall": recall_score(label_list, prediction_list,  mode='strict', scheme=IOB2, average= 'micro')})
        wandb.log({"precision": precision_score(label_list, prediction_list,  mode='strict', scheme=IOB2, average= 'micro')})
        ray.shutdown()
        


if __name__ == "__main__":
    
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Hello this space belogs to NAME!')
    parser.add_argument('--csv_file', required=True, help='Path to the input CSV file')
    parser.add_argument('--csv_file_output', required=True, help='Path to the output CSV file')
    #check if the output file exists
    global args
    args = parser.parse_args()
    wandb.agent(sweep_id, evaluation, count=100)
    pd.read_csv (args.csv_file_output).to_excel ('output.xlsx', index = None, header=True) 
    



