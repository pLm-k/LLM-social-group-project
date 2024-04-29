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
from llm_attacks.minimal_gcg.opt_utils import token_gradients, sample_control, get_logits, target_loss
from llm_attacks.minimal_gcg.opt_utils import load_model_and_tokenizer, get_filtered_cands
from llm_attacks.minimal_gcg.string_utils import SuffixManager, load_conversation_template
from llm_attacks import get_nonascii_toks
from livelossplot import PlotLosses # pip install livelossplot
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.metrics import classification_report
#from seqeval.metrics import classification_report
#from seqeval.scheme import IOB2
# Set the random seed for NumPy
np.random.seed(20)
# Set the random seed for PyTorch
torch.manual_seed(20)


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
        return "++++empty+++++ " + generated_text

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

    
def inference(args):
    if os.path.isfile(args.csv_file_output):
        os.remove(args.csv_file_output)
    comments = read_csv(args.csv_file, 'Comment')
    print(comments)
    #llm = LLM(model="TheBloke/Llama-2-70B-chat-AWQ",quantization="awq", tensor_parallel_size=torch.cuda.device_count())
    llm = LLM(model="TheBloke/Llama-2-70B-Chat-fp16", tensor_parallel_size=torch.cuda.device_count())
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
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens = len(max(comments, key=len)))
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
        


    
def convert_to_bio_format(seq_array):
    print("-----------------")
    print(seq_array)
    count =0
    for seq in seq_array:
        print(seq.split())
        count += 1
    # Apply postprocessing to each element in the array
    seq_array = postprocessing(seq_array)
    split_arrays = np.core.defchararray.split(seq_array)
    seq_array = np.concatenate(split_arrays)
    #print(seq_array)
    print(">>>>>>> seq_array", count)
    # Initialize BIO sequences
    bio_sequence = np.full_like(seq_array, "O", dtype='U1')
    #print(">>>>>> bio_sequence", bio_sequence)
    # Initialize entity flag
    inside_entity = False
    
    # Find the start and end indices of the gold entity
    for i, token in enumerate(seq_array):
        #print(token)
        if token.startswith("@@") and token.endswith("##"):
            #print(1)
            bio_sequence[i] = "B"
            inside_entity = False

        elif token.startswith("@@"):
            bio_sequence[i] = "B"
            inside_entity = False
            next_at_index = -1
            next_hash_index = -1
    
            # Find the index of the next token starting with @@
            for j, t in enumerate(seq_array[i + 1:]):
                if t.startswith("@@"):
                    next_at_index = i + 1 + j
                    break
    
            # Find the index of the next token ending with ##
            for j, t in enumerate(seq_array[i + 1:]):
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


def evaluation(args):
    label = read_csv(args.csv_file, 'Sentence annotation based on annotation aggrement')
    pre = read_csv(args.csv_file_output, 'Output')
    label = postprocessing(np.array(label))
    pre = postprocessing(np.array(pre))

    prediction_list = []
    label_list =[]
    count = 0
    for l, p in zip(label, pre):
        if len(l.split(" ")) == len(p.split(" ")):
            prediction_list.append(l)
            label_list.append(p)
        else:
            count += 1
            # print("------------------------------------------")
            # print(">>>>label: \n", l)
            # print(l.split(" "))
            # print ("\n >>>prediction: \n", p)
            # print(p.split(" "))
    print(classification_report(convert_to_bio_format(np.array(label_list)).tolist(), convert_to_bio_format(np.array(prediction_list)).tolist()))

if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Hello this space belogs to NAME!')
    parser.add_argument('--csv_file', required=True, help='Path to the input CSV file')
    parser.add_argument('--csv_file_output', required=True, help='Path to the output CSV file')
    #check if the output file exists
    args = parser.parse_args()
    inference(args)
    evaluation(args)
    pd.read_csv (args.csv_file_output).to_excel ('output.xlsx', index = None, header=True) 
