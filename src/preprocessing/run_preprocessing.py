import preprocessing_functions as dp
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os


# Get the current working directory
cwd = os.getcwd()

# Create the path to the csv file
csv_file_path = os.path.join(cwd, 'data', 'raw', 'Recipes1000.CSV')

# Create the paths to save locations
save_path_processed_recipes = os.path.join(cwd, 'data', 'processed', 'df_recipes.pkl')

save_path_custom_stopwords = os.path.join(cwd, 'data', 'processed', 'stopword_generation', 'df_custom_stopwords.pkl')

save_path_processed_recipes_no_stopwords =  os.path.join(cwd, 'data', 'processed', 'df_recipes_no_stopwords.pkl')



# Alternative path
#csv_file_path = "data\\raw\\Recipes1000.csv"

df_recipes = pd.read_csv(csv_file_path)

# Reduce dataframe for early development to improve iteration speed
#df_recipes = df_recipes.head(10)

# Determine if recipes are in english by looking at the description
df_recipes["english"] = df_recipes["description"].apply(dp.language_detection)

# Filter df_recipes to only include english recipes
df_recipes = df_recipes[df_recipes['english'] == True]
# reset index of df_recipes
df_recipes.reset_index(drop=True, inplace=True)

# Preprocess all text columns
for column in ["name", "description", "ingredients", "steps"]:

    df_recipes[column] = df_recipes[column].astype(str)

    df_recipes[f"preprocessed_{column}"] = df_recipes[column].apply(dp.keep_only_words) \
        .apply(dp.tokenize_to_list) \
        .apply(dp.filter_language) \
        
    df_recipes[f"token_tag_{column}"] = df_recipes[f"preprocessed_{column}"].apply(dp.token_and_tag_to_list)

    df_recipes[f"preprocessed_{column}"] = df_recipes[f"preprocessed_{column}"].apply(dp.lemmatize_to_list) \
        .apply(dp.non_stopword_token_to_list) \
        .apply(dp.upper_tokens_to_list)

# Save dataframe as pickle file to preserver python data types
df_recipes.to_pickle(save_path_processed_recipes)
# Alternative path
#df_recipes.to_pickle("data\\processed\\df_recipes.pkl")


# Generate custom stopwords
url = "https://en.wikipedia.org/wiki/Cooking_weights_and_measures"

response = requests.get(url)
if response.status_code == 200:
    content = response.text

soup = BeautifulSoup(content, 'html.parser')
text = soup.get_text()
text = re.sub(r'\t', '', text)

df_measures = pd.DataFrame(columns=['content'], data=[line for line in text.split('\n') if line.strip()])

# Preprocess all text columns
for column in ["content"]:

    df_measures[column] = df_measures[column].astype(str)

    df_measures[f"preprocessed_{column}"] = df_measures[column].apply(dp.keep_only_words) \
        .apply(dp.tokenize_to_list) \
        .apply(dp.filter_language) \
        
    df_measures[f"token_tag_{column}"] = df_measures[f"preprocessed_{column}"].apply(dp.token_and_tag_to_list)

    df_measures[f"preprocessed_{column}"] = df_measures[f"preprocessed_{column}"].apply(dp.lemmatize_to_list) \
        .apply(dp.non_stopword_token_to_list) \
        .apply(dp.upper_tokens_to_list)

# Save dataframe as pickle file to preserver python data types
df_measures.to_pickle(save_path_custom_stopwords)
# Alternative path
#df_measures.to_pickle("data\\processed\\stopword_generation\\df_custom_stopwords.pkl")


# Get the custom stopwords from df_measures
custom_stopwords = set()
for words in df_measures['preprocessed_content']:
    for word in words:
        custom_stopwords.add(word)

additional_stopwords = [ 'TABLESPOON', 'NONE', 'FRESH', 'GRAM', 'GROUND', 'POUND', 'POWDER', 'CHOPPED', 'RED', 'MEDIUM', 'VIRGIN', 'SPRAY', 'PINCH', 'BONELESS', 
                        'SLICE', 'SKINLESS', 'MILLILITER', 'BABY', 'PACKAGE', 'BAG', 'MIX', 'BUNCH', 'INCH', 'SWEET', 'EXTRACT', 'PURE', 'HEAD', 'CUT', 'JAR', 'SPREAD', 'SQUASH',
                        'PLAIN', 'KOSHER', 'REMOVE', 'LET', 'BE', 'SET', 'F', 'ADD', 'MAKE', 'SEE', 'TAKE', 'CONTINUE', 'BRING', 'COOK', 'GET', 'MAKE', 'KEEP', 'YES',
                        'SALT', 'PEPPER', 'OIL', 'WATER', 'ONION', 'OLIVE', 'SUGAR', 'BUTTER', 'SAUCE', 'JUICE', 'CREAM', 'LARGE', 'BLACK', 'WHITE', 'GREEN', 'BROWN', 'WHOLE',
                        'PURPOSE', 'FREE', 'DRIED', 'FAT', 'SLICED', 'BAKING', 'UNSALTED', 'FROZEN', 'CRUSHED', 'TASTE', 'BELL', 
                        'LIGHT', 'EXTRA', 'YELLOW', 'GRANULATED', 'ORANGE', 'BEST', 'REDUCED', 'HOT', 'DRY', 'LOW', 'HALF', 'WISH', 'THIN', 'SHARP', 'DARK', 'HANDFUL', 'HEAVY', 'REAL', 'DRAINED', 'CANT', 'BELIEVE', 'SMALL', 'LEFTOVER', 'CUBE', 'OPTIONAL', 'STYLE', 'BEATEN', 'CANNED',
                        'CONDENSED', 'POWDERED', 'FRESHLY', 'PEELED', 'BAY', 'SEA', 'FINELY', 'DIVIDED', 'SMOKED', 'NATURAL', 'LEAF', 'ASIDE', 'BOWL', 'DISH', 'COMBINE', 'MADE', 'SERVE', 'GREAT', 'DELICIOUS', 'PLACE', 'ADD', 'MIXTURE', 'CUP', 'TOP' 
                        ]



for word in additional_stopwords:
    custom_stopwords.add(word)

# Function to remove custom stopwords from a list of words
def remove_custom_stopwords(word_list):
    return [word for word in word_list if word not in custom_stopwords]


# Preprocess all text columns in df_recipes
for column in ["name", "description", "ingredients", "steps"]:
    df_recipes[column] = df_recipes[column].astype(str)

    df_recipes[f"preprocessed_{column}"] = df_recipes[column].apply(dp.keep_only_words) \
        .apply(dp.tokenize_to_list) \
        .apply(dp.filter_language)
        
    df_recipes[f"token_tag_{column}"] = df_recipes[f"preprocessed_{column}"].apply(dp.token_and_tag_to_list)

    df_recipes[f"preprocessed_{column}"] = df_recipes[f"preprocessed_{column}"].apply(dp.lemmatize_to_list) \
        .apply(dp.non_stopword_token_to_list) \
        .apply(dp.upper_tokens_to_list) \
        .apply(remove_custom_stopwords)  # remove the custom stopwords
    
    # Remove all list entries in the 'preprocessed_' columns that are 'NONE'
    for lst in df_recipes[f"preprocessed_{column}"]:
        # remove all 'NONE' entries from lst
        while 'NONE' in lst:
            lst.remove('NONE')

    # Remove all ('NONE','NN') entries in a 'token_tag_' column
    for column in ["token_tag_name", "token_tag_description", "token_tag_ingredients", "token_tag_steps"]:
        for lst in df_recipes[column]:
            # remove all ('NONE','NN) entries from lst
            while ('NONE','NN') in lst:
                lst.remove(('NONE','NN'))

# Save df_recipes
df_recipes.to_pickle(save_path_processed_recipes_no_stopwords)
# Alternative path
#df_recipes.to_pickle("data\\processed\\df_recipes_no_stopwords.pkl")