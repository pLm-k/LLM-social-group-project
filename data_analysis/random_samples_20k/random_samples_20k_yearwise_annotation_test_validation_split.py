import pandas as pd
import time
import argparse
import random as rd
import os

#random seed for sampling
random_seed = int.from_bytes(os.urandom(4), byteorder='big')
rd.seed(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default=None, type=str, required=True, help='path/data_analysis')
args = parser.parse_args()

process_start = time.time()

years = ['2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
months = ['01', '02', '03', '04', '05','06','07','08','09','10','11','12']

#data structures for collecting results
subreddits = ['politics', 'ukpolitics', 'Libertarian', 'Economics', 'worldpolitics']
samples_total = {}
test_samples = {}
validation_samples = {}
annotation_samples = {}

samples = pd.DataFrame(columns=['body_cleaned','id','subreddit'])

for subreddit in subreddits:
    samples_total[subreddit] = []
    test_samples[subreddit] = samples.copy()
    validation_samples[subreddit] = samples.copy()
    annotation_samples[subreddit] = samples.copy()

#set number of samples per year
n_samples_year = 334
n_samples_year_split = 34

#go over all years
for year in years:
    start_time = time.time()
    
    # collect comment per subreddit over year
    comments_year = {}
    for subreddit in subreddits:
            comments_year[subreddit] = []

    for month in months:
        data = pd.read_csv(args.dir+ '/data_preproc/' + year + '/comments_' + year + '-' + month + '_preproc.csv', sep='\t')

        for row in data.itertuples(index=False):
            #consider only if subreddit is in list
            subreddit = row.subreddit
            if subreddit not in subreddits:
                continue
            comments_year[subreddit].append(row)
    
    # sample from data of year (per subreddit) and collect
    for subreddit in subreddits:
        samples_year_subreddit = rd.sample(comments_year[subreddit], n_samples_year)
        samples_total[subreddit] += samples_year_subreddit  

    print(f"compute of {year} took {time.time() - start_time} seconds")
    print("-----------------------------------------------------------------------------")

#split into annotation, test and validation sets
for subreddit in subreddits:
    #get samples by year (of one subreddit)
    samples_by_year = [samples_total[subreddit][i * n_samples_year: (i + 1) * n_samples_year] for i in range(len(years))]
    
    #sample for annotation and remove samples from available samples
    for i in range(len(years)):
        annotation_samples_year = rd.sample(samples_by_year[i], n_samples_year_split)
        for sample in annotation_samples_year:
            samples_by_year[i].remove(sample)
        annotation_samples[subreddit] = pd.concat([annotation_samples[subreddit], pd.DataFrame(annotation_samples_year, columns=samples.columns)],ignore_index=True)
    
    #sample for test and remove samples from available samples
    for i in range(len(years)):
        test_samples_year = rd.sample(samples_by_year[i], n_samples_year_split)
        for sample in test_samples_year:
            samples_by_year[i].remove(sample)
        test_samples[subreddit] = pd.concat([test_samples[subreddit], pd.DataFrame(test_samples_year, columns=samples.columns)],ignore_index=True)
    
    #take rest of samples for validation
    for i in range(len(years)):
        validation_samples[subreddit] = pd.concat([validation_samples[subreddit], pd.DataFrame(samples_by_year[i], columns=samples.columns)],ignore_index=True)

    annotation_samples[subreddit].to_csv(args.dir+f'/random_samples_20k/annotation/random_samples_annotation_{subreddit}.csv', index=False, sep='\t')
    test_samples[subreddit].to_csv(args.dir+f'/random_samples_20k/test/random_samples_test_{subreddit}.csv', index=False, sep='\t')
    validation_samples[subreddit].to_csv(args.dir+f'/random_samples_20k/validation/random_samples_validation_{subreddit}.csv', index=False, sep='\t')

print("-----------------------------------------------------------------------------")
print('total time: ' + str(time.time() - process_start) +'s')

print(f'random seed: {random_seed}')