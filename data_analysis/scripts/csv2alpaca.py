import pandas as pd
import json
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default=None, type=str, required=True, help='path/data')
parser.add_argument('--num_data', default=sys.maxsize, type=int, required=False, help='how many datapoints to consider')
args = parser.parse_args()

subreddits = ['politics', 'ukpolitics', 'Libertarian', 'Economics', 'worldpolitics']

prompt = """###Definition: Social groups are defined by two or more individuals who share some common characteristics. Commonalities shared by a social group include, for example, race, nationality, ethnicity, religion, gender, sexual orientation, socio-economic status, migration status, profession, family ties, and organizational and institutional ties.

    ###Instruction: Annotate the text after ###input:, only add @@ and ## dont remove or add any additional symbols.

    ###input: """

for subreddit in subreddits:
    data = pd.read_csv(args.dir+ f'/random_samples_test_{subreddit}_annotated_gold.csv', sep='\t')
    out_data = []
   
    for i, row in enumerate(data.itertuples(index=False)):
        if i >= args.num_data:
            break
        
        out_data.append({"instruction": f'[INST] {prompt}<annotation> {row.body_cleaned} </annotation> [/INST]', "output": f'<annotation> {row.gold_annotation} </annotation>', "input":""})

    with open(f'random_samples_test_{subreddit}_alpaca.json', "w") as json_file:
        json.dump(out_data, json_file, indent=4)

            
