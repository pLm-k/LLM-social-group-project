import pandas as pd
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default=None, type=str, required=True, help='path/data_analysis')
args = parser.parse_args()

process_start = time.time()

years = ['2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
months = ['01', '02', '03', '04', '05','06','07','08','09','10','11','12']

comments_subreddit_total = {}

#go over all years
for year in years:
    comments_subreddit_year = {}
    start_time = time.time()
    for month in months:
        data = pd.read_csv(args.dir+ '/data/' + year + '/comments_' + year + '-' + month + '.csv', sep='\t')
        
        for row in data.itertuples():
            #skip if comment was empty
            comment = row.body_cleaned
            if pd.isna(comment):
                continue

            subreddit = row.subreddit
            
            if subreddit not in comments_subreddit_year.keys():
                comments_subreddit_year[subreddit] = 0
            
            comments_subreddit_year[subreddit] += 1

    print("-----------------------------------------------------------------------------")
    print(f"compute of {year} took {time.time() - start_time} seconds")
    print(f"total number of comments {sum(comments_subreddit_year.values())}")
    
    #print sorted csv for year
    comments_subreddit_year_sorted_list = sorted(comments_subreddit_year.items(), key=lambda x: x[1], reverse=True)
    year_df = pd.DataFrame(comments_subreddit_year_sorted_list, columns=['subreddit', 'n_comments'])
    year_df.to_csv(args.dir+f'/results/result_comments_{year}.csv', index=False, sep='\t')

    #update data over all years
    for subreddit, n_comments in comments_subreddit_year.items():
        if subreddit not in comments_subreddit_total.keys():
            comments_subreddit_total[subreddit] = 0

        comments_subreddit_total[subreddit] += n_comments
    
#print sorted csv over all years
comments_subreddit_total_sorted_list = sorted(comments_subreddit_total.items(), key=lambda x: x[1], reverse=True)
total_df = pd.DataFrame(comments_subreddit_total_sorted_list, columns=['subreddit', 'n_comments'])
total_df.to_csv(args.dir+'/results/result_comments_total.csv', index=False, sep='\t')

print("-----------------------------------------------------------------------------")
print('total time: ' + str(time.time() - process_start) +'s')


        
    
    
    
    
    




            
           


