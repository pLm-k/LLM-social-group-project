import textstat as tex
import pandas as pd
import time
import argparse
import heapq as hq

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default=None, type=str, required=True, help='path/data_analysis')
parser.add_argument('--num_longest', default=10, type=int, required=False, help='num of top longest comments')
args = parser.parse_args()

process_start = time.time()

years = ['2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
months = ['01', '02', '03', '04', '05','06','07','08','09','10','11','12']

#setup prio queue for getting longest comments (Format (comment,year-month,id))
longest_comments_hq = []
for q_element in range(0,args.num_longest):
    hq.heappush(longest_comments_hq, (0,('','','ffff')))

rows_subreddit = {}
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
            id = row.id
            subreddit = row.subreddit
            
            if subreddit not in rows_subreddit.keys():
                rows_subreddit[subreddit] = {'num_comments':0, 'num_sentences':0,'avg_num_sentences':0,'max_num_sentences':0,
                                                  'num_words':0,'avg_num_words':0, 'max_num_words':0, 'avg_words_sentence':0}

            #get sentences in comment
            rows_subreddit[subreddit]['num_comments'] += 1
            num_sentences_comment = tex.sentence_count(comment)
            rows_subreddit[subreddit]['num_sentences'] += num_sentences_comment
            if num_sentences_comment > rows_subreddit[subreddit]['max_num_sentences']:
                rows_subreddit[subreddit]['max_num_sentences'] = num_sentences_comment

            #get words in comment
            num_words_comment = tex.lexicon_count(comment, removepunct=True)
            rows_subreddit[subreddit]['num_words'] += num_words_comment
            if num_words_comment > rows_subreddit[subreddit]['max_num_words']:
                rows_subreddit[subreddit]['max_num_words'] = num_words_comment

            #get comment if long enough (we pushed n elements with prio 0 at the start!)
            hq.heappushpop(longest_comments_hq, (num_words_comment, (comment, year + '-' + month, id)))

        print('compute time: '+ str(time.time()-start_time) +'s')
        print('computed     ' + year + '-' + month)
        print('---------------------------------')

for subreddit in rows_subreddit.keys():
    rows_subreddit[subreddit]['avg_num_sentences'] = rows_subreddit[subreddit]['num_sentences']/rows_subreddit[subreddit]['num_comments']
    rows_subreddit[subreddit]['avg_num_words'] = rows_subreddit[subreddit]['num_words']/rows_subreddit[subreddit]['num_comments']
    rows_subreddit[subreddit]['avg_words_sentence'] = rows_subreddit[subreddit]['num_words']/rows_subreddit[subreddit]['num_sentences']
#print df describing all years
results = pd.DataFrame(list(rows_subreddit.values()), index=list(rows_subreddit.keys()), columns=['num_comments', 'num_sentences','avg_num_sentences','max_num_sentences',
                                                  'num_words','avg_num_words', 'max_num_words', 'avg_words_sentence']).reset_index()    
results = results.rename(columns={'index': 'subreddit'})
results = results.sort_values(by='num_comments', ascending=False)
results.to_csv(args.dir+'/results/results_analysis_per_subreddit.csv', index=False, sep='\t')

print('total time: ' + str(time.time() - process_start) +'s')

#print longest sentences to file
with open(args.dir+'/results/longest_sentences_per_subreddit.txt', 'w', encoding='utf-8') as file:
    # print sentences in descending order
    while len(longest_comments_hq) > 0:
        longest_comment = hq.heappop(longest_comments_hq)
        print(longest_comment[1][0], file=file)
        print('-----------------------------------------------------------------------------------------------------', file=file)
        print('id: ' + longest_comment[1][2] + '   Posted in: ' + longest_comment[1][1] + '   num_words: ' + str(longest_comment[0]), file=file)
        print('-----------------------------------------------------------------------------------------------------', file=file)
        print('', file=file)


        
    
    
    
    
    




            
           


