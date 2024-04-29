import pandas as pd

# Load your CSV file
file_path = '50_samples(2).csv' # Update this to the path of your CSV file
data = pd.read_csv(file_path)

#init dataframe column for sentiment annotations
column_name='sentiment_annotator_1'
data[column_name] = None

# Function to collect sentiment annotations from the user
def collect_user_sentiment_annotations(data):
    for index, row in data.iterrows():
        print(f"Row {index+1}/{len(data)}")
        print("Comment:", row['Comment'])
        print("Annotation Agreement:", row['annotation aggrement'])
        
        # Prompt user for sentiment annotation
        user_input = input("Enter your sentiment annotation for this comment: ")
        print("INPUT:", user_input)
        # Update the dataframe with the user input
        data.at[index, column_name] = user_input
    return data

# Collect user inputs for sentiment annotations
updated_data = collect_user_sentiment_annotations(data)

# Save the to a CSV file
updated_file_path = 'annotated' + file_path
updated_data.to_csv(updated_file_path, index=False)


