import pandas as pd
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default=None, type=str, required=True, help='path/data_analysis')
args = parser.parse_args()

process_start = time.time()

years = ['2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
months = ['01', '02', '03', '04', '05','06','07','08','09','10','11','12']

#setup dict for special symbol counting
special_symbols = ['!', '@', '#', '$', '%', '&', '*', '!!', '@@', '##', '$$', '%%', '&&', '**']
special_symbols_dict = {}
for symbol in special_symbols:
    special_symbols_dict[symbol] = 0

#go over all years
for year in years:
    #go over all months of a year
    for month in months:
        start_time = time.time()
        data = pd.read_csv(args.dir+ '/data/' + year + '/comments_' + year + '-' + month + '.csv', sep='\t')
        print('read time: '+ str(time.time()-start_time) +'s')
        
        start_time = time.time()
        for row in data.itertuples(index=False):
            #skip if comment was empty
            comment = row.body_cleaned
            if pd.isna(comment):
                continue
            
            #get number of special symbols in comment
            for symbol in special_symbols:
                special_symbols_dict[symbol] += comment.count(symbol)

        print('compute time: '+ str(time.time()-start_time) +'s')
        print('computed     ' + year + '-' + month)
        print('---------------------------------')

print('total time: ' + str(time.time() - process_start) +'s')

#print num of appearences of special symbols
special_symbols_df = pd.DataFrame(list(special_symbols_dict.items()), columns=['symbol','num_appeared'])
special_symbols_df.to_csv(args.dir+'/results/results_special_symbols.csv', index=False, sep='\t')


        
    
    
    
    
    




            
           


