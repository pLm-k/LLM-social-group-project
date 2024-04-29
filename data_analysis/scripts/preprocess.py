import pandas as pd
import time
import argparse
import re

def remove_symbols(comment):
    regEx = r"[^a-zA-Z0-9,.?!'\s]"
    tmp_string = re.sub(regEx, "", comment) #remove symbols
    return re.sub(r'\s+', ' ', tmp_string).strip() #remove consecutive whitespaces


parser = argparse.ArgumentParser()
parser.add_argument('--dir', default=None, type=str, required=True, help='path/data_analysis')
args = parser.parse_args()

process_start = time.time()

years = ['2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'] #!--------------------------------set desired years--------------------------!
months = ['01', '02', '03', '04', '05','06','07','08','09','10','11','12']



#go over all years
for year in years:
    start_time = time.time()
    
    for month in months: 
        data = pd.read_csv(args.dir+ '/data/' + year + '/comments_' + year + '-' + month + '.csv', sep='\t')
        rows_month = []
        
        for row in data.itertuples(index=False): 
            comment = row.body_cleaned
            #consider only if comment not empty
            if pd.isna(comment):
                continue
            
            #remove special chars
            comment = remove_symbols(comment) 
            words = comment.split()
            #consider only if comment has at least 5 words of length >= 3 and at least 10% unique words
            words_longer_than_3 = [word for word in words if len(word) >= 3]
            if len(words_longer_than_3) < 5 or len(set(words))/len(words) < 0.1:
                continue

            rows_month.append({'body_cleaned':comment, 'id':row.id, 'subreddit':row.subreddit})
        
        df_month = pd.DataFrame(rows_month, columns=['body_cleaned','id','subreddit'])
        df_month.to_csv(args.dir+f"/data_preproc/{year}/comments_{year}-{month}_preproc.csv", index=False, sep='\t')

    print(f"processing of {year} took {time.time() - start_time} seconds")
    print("-----------------------------------------------------------------------------")

print("-----------------------------------------------------------------------------")
print('total time: ' + str(time.time() - process_start) +'s')





        
    
    
    
    
    




            
           


