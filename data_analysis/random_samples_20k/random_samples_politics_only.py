import pandas as pd
import time
import argparse
import random as rd
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default=None, type=str, required=True, help='path/data_analysis')
args = parser.parse_args()

process_start = time.time()

years = ['2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
months = ['01', '02', '03', '04', '05','06','07','08','09','10','11','12']

#set number of samples per year
n_samples_year = 167

samples = pd.DataFrame(columns=['body_cleaned','id','subreddit'])

#go over all years
for year in years:
    start_time = time.time()
    n_samples_months = Counter([rd.randint(1, 12) for _ in range(n_samples_year)])
    
    for month in months:
        n_samples = n_samples_months[int(month)]
        if n_samples == 0:
            continue
        
        data = pd.read_csv(args.dir+ '/data/' + year + '/comments_' + year + '-' + month + '.csv', sep='\t')
        politics_comments_month = []

        for row in data.itertuples(index=False):
            comment = row.body_cleaned
            #consider only if subreddit is politics and the comment is not empty
            if row.subreddit != 'politics':
                continue

            politics_comments_month.append(row)
        
        samples_month = rd.sample(politics_comments_month, n_samples)
        samples = pd.concat([samples, pd.DataFrame(samples_month, columns=samples.columns)], ignore_index=True)

    print(f"compute of {year} took {time.time() - start_time} seconds")
    print("-----------------------------------------------------------------------------")

samples.to_csv(args.dir+'/random_samples_preproc.csv', index=False, sep='\t')

print("-----------------------------------------------------------------------------")
print('total time: ' + str(time.time() - process_start) +'s')





        
    
    
    
    
    




            
           


