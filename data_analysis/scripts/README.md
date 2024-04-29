# Analysis Scripts
Provide path to data_analysis folder using --dir.</br>
Set specific subreddits in script. </br>
Note: Scripts assume preprocessed data! </br>

* **preprocess**: Preprocess dataset (data from data folder, saved in data_preproc folder).
* **IAA**: Inter annotator agreement (for running adapt test method in l.222).
* **analyze_data_by_(sub|year)**: Do data analysis on the data in data folder (group by subreddit or year).
* **special_symbol_analysis**: Occurences of specific symbols of data in data folder (change symbols in script).
* **comments_subreddit**: Number of comments per subreddit over all years (this functionality is also in analyze_data).
* **csv2alpace**: Transform csv into json file for fine-tuning.
