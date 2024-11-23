import numpy as np  # working with arrays and linear algebra (numerical python)
import pandas as pd  # data manipulation and analysis
import nltk  # text processing library
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # sentiment analysis
from textblob import TextBlob  # text processing and sentiment analysis
from wordcloud import WordCloud  # for generating word clouds
import seaborn as sns  # data visualization
import re  # regular expressions
import matplotlib.pyplot as plt  # data visualization
import cufflinks as cf  # charts can be created on dataframes
import plotly.graph_objs as go  # Plotly graph objects
from plotly.subplots import make_subplots  # for creating subplots
import warnings  # for managing warnings
from plotly.offline import plot
from transformers import pipeline

# Suppress warnings
warnings.filterwarnings("ignore")

# Adjust Pandas display settings
pd.set_option("display.max_columns", None)

# Initialize Cufflinks for offline mode
cf.go_offline()

# Download necessary NLTK data
nltk.download('vader_lexicon')

print("Setup complete! Libraries are imported correctly.")

#importing dataset
df = pd.read_csv('/Users/sajajalil/Desktop/amazon.csv')
print(df.head)

#sorting values
df = df.sort_values("wilson_lower_bound", ascending=False)
df.drop('Unnamed: 0', inplace=True, axis=1)
df.head()

#handeling missing values
def missing_values_analysis(df):
    #identifying columns with missing values
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    #identifying missing values in columns
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=True)
    #calculating the missing value ratio
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)
    # concat the two columns in a data frame one for the missing values (Missing Values) and one for the missing value percentage (Ratio).
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df

#anaylsing dataframe
def check_dataframe(df,head=5, tail=5):
    print("SHAPE".center(82,'~'))
    print("Rows{}".format(df.shape[0]))
    print("Columns{}".format(df.shape[1]))
    print("TYPES".center(82,'~'))
    print(df.dtypes)
    print("".center(82,'~'))
    print(missing_values_analysis(df))
    print("DUPLICATED VALUES".center(82,'~'))
    print(df.duplicated().sum())
    print("QUANTITIES".center(82,'~'))
    numeric_df = df.select_dtypes(include=[np.number])
    print(numeric_df.quantile([0, 0.05, 0.5, 0.95, 0.99, 1]).T)

check_dataframe(df)

#checking unique values
def check_class(dataframe):
    nunique_df = pd.DataFrame({'Variable': dataframe.columns,
                              'Classes': [dataframe[i].nunique() for i in dataframe.columns]})
    nunique_df = nunique_df.sort_values('Classes', ascending=False)
    nunique_df = nunique_df.reset_index(drop=True)
    return nunique_df

print(check_class(df))

#categorical value analysis
constraints = ['#B34D22','#EBE00C','#1FEB0C','#0C92EB','#EB0CD5']

def categorical_variable_summary(df, column_name, constraints=None):
    # Create the subplots with the correct 'type' for bar and pie charts
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=('Countplot', 'Percentage'),
        specs=[[{"type": "xy"}, {"type": "domain"}]]
    )

    # Add the bar plot for countplot
    bar_trace = go.Bar(
        x=df[column_name].value_counts().index.tolist(),
        y=df[column_name].value_counts().values.tolist(),
        showlegend=False,
        marker=dict(color=constraints, line=dict(color='#DBE6EC', width=1))
    )

    # Add the bar trace to the subplot (specifying row=1 and col=1)
    fig.add_trace(bar_trace, row=1, col=1)

    # Add the pie chart for percentage
    fig.add_trace(
        go.Pie(
            labels=df[column_name].value_counts().keys(),
            values=df[column_name].value_counts().values,
            textfont=dict(size=18),
            textposition='auto',
            showlegend=False,
            name=column_name,
            marker=dict(colors=constraints)
        ),
        row=1, col=2
    )

    # Update layout with title and layout configuration
    fig.update_layout(
        title={"text": column_name, 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        template='plotly_white'
    )

    # Display the plot
    plot(fig)

#categorical_variable_summary(df,'overall')

#cleaning data
df.reviewText.head()
#This line retrieves the value at the 2032nd position (remember Python is 0-indexed) from the reviewText column and assigns it to the variable review_example.
review_example = df.reviewText[2031]
#cleaning data using regular expressions
review_example = re.sub("[^a-zA-Z]",'',review_example)
review_example

#convert all text to lower case to not consider uppercase words as different
review_example = review_example.lower().split()
rt = lambda x: re.sub("[^a-zA-Z]",' ',str(x))
df["reviewText"] = df["reviewText"].map(rt)
df["reviewText"] = df["reviewText"].str.lower()
print(df.head())

#sentiment analysis
#polaity indicates the mood, whether its pos,neg or neutral and returns a value between 0 and 1, the closer to one is more positive, the closer to 0 is more negative
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Adding polarity and subjectivity using TextBlob
df[['polarity', 'subjectivity']] = df['reviewText'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))

analyzer = SentimentIntensityAnalyzer()

# Iterate over the 'reviewText' column and calculate sentiment
for index, row in df.iterrows(): 
    text = row['reviewText']
    score = analyzer.polarity_scores(text)
    
    # Get negative, neutral, and positive scores
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    
    # Assign sentiment based on scores
    if neg > pos:
        df.loc[index, 'sentiment'] = 'Negative'
    elif pos > neg:
        df.loc[index, 'sentiment'] = 'Positive'
    else:
        df.loc[index, 'sentiment'] = 'Neutral'

print(df[['reviewText', 'sentiment']].head())

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['neg'] > score['pos']:
        return 'Negative'
    elif score['pos'] > score['neg']:
        return 'Positive'
    else:
        return 'Neutral'

df['sentiment'] = df['reviewText'].apply(get_sentiment)



df[df['sentiment'] == 'Positive'].sort_values("wilson_lower_bound",ascending=False).head(5)
categorical_variable_summary(df,'sentiment')

