from vllm import LLM,SamplingParams
import time
import pandas as pd
import argparse
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="your_project",
    
    # track run metadata
    config={
    "model": "TheBloke/Llama-2-70B-Chat-fp16",
    "dataset": "Reddit-Politosphere-2008-jan-cleaned",
    }
)

#arguments when calling the inference script
parser = argparse.ArgumentParser()
parser.add_argument('--infile', default=None, type=str, required=True, help='path to input csv')
parser.add_argument('--outfile', default=None, type=str, required=True, help='path to output csv')
args = parser.parse_args()



#load csv
read_start = time.time()

in_frame = pd.read_csv(args.infile, sep='\t')

print('---------------------------------')
print('read time: ' + str(time.time() - read_start) + 's')
print('---------------------------------')

prompt = """ Below is an instruction that describes a task. 
    Write a response that appropriately completes the request.
    ### Instruction: The task is to annotate social groups. The output starts with <annotation> and ends with <\annotation> and number of words in the output are equal to the number of words in the input.  Do annotation for social groups starting with @@ and ending with ##. Social groups are defined by two or more individuals who share some common characteristics. Commonalities shared by a social group include, for example, race, nationality, ethnicity, religion, gender, sexual orientation, socio-economic status, migration status, profession, family ties, and organizational and institutional ties. ###Examples: input: Because when the middle class does well, the poor have a ladder up and the wealthy still do very well. output: <annotation> Because when @@the middle class## does well, @@the poor ##have a ladder up and @@the wealthy ##still do very well. </annotation> input: hillary is stating publicly that her true allegiance is to the ultra-zionists . not the democratic party , not the usa , not anyone. output: <annotation> hillary is stating publicly that her true allegiance is to @@the ultra-zionists ##. not @@ the democratic party## , not @@the usa ##, not anyone. </annotation> input: President Biden said: near-record unemployment for Black and Hispanic workers. 
    output: <annotation> President Biden said: near-record unemployment for @@Black and Hispanic workers##.</annotation> input: Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership 
    output: <annotation> Mr. Biden thanked Mr. Chaves, who was elected last year, for his leadership </annotation>
    ###input:  """

#prepare prompts and result dicts for output
prompts = []
results = []
for row in in_frame.itertuples(index=False):
    comment = row.body_cleaned
    #filter out empty comments
    if pd.isna(comment):
        continue
    prompts.append(f"{prompt}{comment}")
    results.append({'body_cleaned':comment, 'id':row.id, 'generated_text':''})



#load model
model_load_start = time.time()

sampling_params = SamplingParams(temperature=0.5, max_tokens=100)
llm = LLM(model="TheBloke/Llama-2-70B-Chat-fp16", tensor_parallel_size=4)  # Name or path of your model

print('---------------------------------')
print('load model: ' + str(time.time() - model_load_start) + 's')
print('---------------------------------')



#inference
gen_start = time.time()

outputs = llm.generate(prompts, sampling_params)
for j, output in enumerate(outputs):
    results[j]['generated_text'] = output.outputs[0].text

print('---------------------------------')
print('generation time: ' + str(time.time() - gen_start) + 's')
print('---------------------------------')



write_start = time.time()

out_frame = pd.DataFrame(results)
out_frame.to_csv(args.outfile, index=False, sep='\t')

print('---------------------------------')
print('write output time: ' + str(time.time() - write_start) + 's')
print('---------------------------------')

wandb.finish()