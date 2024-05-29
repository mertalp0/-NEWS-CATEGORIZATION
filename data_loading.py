import pandas as pd
from cleaning import clean_text

def load_and_clean_data(file_name):
    """
    Loads and cleans the data.
    """
    df = pd.read_csv(file_name)
    df['CLEANED_TEXT'] = df['text'].apply(clean_text)
    return df
