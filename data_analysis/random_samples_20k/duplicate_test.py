import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default=None, type=str, required=True, help='path/data_analysis')
args = parser.parse_args()

subreddits = ['politics', 'ukpolitics', 'Libertarian', 'Economics', 'worldpolitics']
test_samples = {}
validation_samples = {}
annotation_samples = {}

ids = set()

for subreddit in subreddits:
    test_samples[subreddit] = pd.read_csv(args.dir + f'/random_samples_20k/test/random_samples_test_{subreddit}.csv', sep='\t')
    validation_samples[subreddit] = pd.read_csv(args.dir + f'/random_samples_20k/validation/random_samples_validation_{subreddit}.csv', sep='\t')
    annotation_samples[subreddit] = pd.read_csv(args.dir + f'/random_samples_20k/annotation/random_samples_annotation_{subreddit}.csv', sep='\t')

for subreddit in subreddits:
    for row in test_samples[subreddit].itertuples(index=False):
        if row.id in ids:
            print('duplicate found!')
        ids.add(row.id)
    
    for row in validation_samples[subreddit].itertuples(index=False):
        if row.id in ids:
            print('duplicate found!')
        ids.add(row.id)
    
    for row in annotation_samples[subreddit].itertuples(index=False):
        if row.id in ids:
            print('duplicate found!')
        ids.add(row.id)

print(f'unique ids: {len(ids)}')
