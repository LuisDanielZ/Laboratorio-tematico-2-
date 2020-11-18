import numpy as np 
import pandas as pd 
import re 
import string
import requests
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, plot, iplot
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# Load data
train = pd.read_csv('data/Corona_NLP_train.csv', encoding = 'latin1')
# Copy  data
df = train.copy()

# Load in test data
test_df = pd.read_csv('data/Corona_NLP_test.csv', encoding = 'latin1')

# Replace nan with 'None'
df['Location'].fillna('None', inplace = True)

# Join stopwords together and set them for use in cleaning function.
", ".join(stopwords.words('english'))
stops = set(stopwords.words('english'))

# Function that cleans tweets for classification.
def process_text(tweet):
    # Remove hyperlinks.
    tweet= re.sub(r'https?://\S+|www\.\S+','',tweet)
    # Remove html
    tweet = re.sub(r'<.*?>','',tweet)
    # Remove numbers
    tweet = re.sub(r'\d+','',tweet)
    # Remove mentions
    tweet = re.sub(r'@\w+','',tweet)
    # Remove punctuation
    tweet = re.sub(r'[^\w\s\d]','',tweet)
    # Remove whitespace
    tweet = re.sub(r'\s+',' ',tweet).strip()
    # Remove stopwords
    tweet = " ".join([word for word in str(tweet).split() if word not in stops])

    return tweet.lower()

# Apply text cleaning function to training and test dataframes.
df['newTweet'] = df['OriginalTweet'].apply(lambda x: process_text(x))
test_df['newTweet'] = test_df['OriginalTweet'].apply(lambda x: process_text(x))

def token_lemma(tweet):
    tk = TweetTokenizer()
    lemma = WordNetLemmatizer()
    tweet = tk.tokenize(tweet)
    tweet = [lemma.lemmatize(word) for word in tweet]
    tweet = " ".join([word for word in tweet])
    return tweet

tweet = df['newTweet']
tweet

df['lemmaTweet'] = df['newTweet'].apply(lambda x: token_lemma(x))

mapper = {}
for i,cat in enumerate(df['Sentiment'].unique()):
        mapper[cat] = i

df['sentiment_target'] = df['Sentiment'].map(mapper)
print(df)

# Some frequent US locations
us_filters = ('New York', 'New York, NY', 'NYC', 'NY', 'Washington, DC', 'Los Angeles, CA',
             'Seattle, Washington', 'Chicago', 'Chicago, IL', 'California, USA', 'Atlanta, GA',
             'San Francisco, CA', 'Boston, MA', 'New York, USA', 'Texas, USA', 'Austin, TX',
              'Houston, TX', 'New York City', 'Philadelphia, PA', 'Florida, USA', 'Seattle, WA',
             'Washington, D.C.', 'San Diego, CA', 'Las Vegas, NV', 'Dallas, TX', 'Denver, CO',
             'New Jersey, USA', 'Brooklyn, NY', 'California', 'Michigan, USA', 'Minneapolis, MN',
             'Virginia, USA', 'Miami, FL', 'Texas', 'Los Angeles', 'United States', 'San Francisco',
             'Indianapolis, IN', 'Pennsylvania, USA', 'Phoenix, AZ', 'New Jersey', 'Baltimore, MD',
             'CA', 'FL', 'DC', 'TX', 'IL', 'MA', 'PA', 'GA', 'NC', 'NJ', 'WA', 'VA', 'PAK', 'MI', 'OH',
             'CO', 'AZ', 'D.C.', 'WI', 'MD', 'MO', 'TN', 'Florida', 'IN', 'NV', 'MN', 'OR','LA', 'Michigan',
             'CT', 'SC', 'OK', 'Illinois', 'Ohio', 'UT', 'KY', 'Arizona', 'Colorado')

# Various nation's frequent locations
uk_filters = ('England', 'London', 'london', 'United Kingdom', 'united kingdom',
              'England, United Kingdom', 'London, UK', 'London, England',
              'Manchester, England', 'Scotland, UK', 'Scotland', 'Scotland, United Kingdom',
              'Birmingham, England', 'UK', 'Wales')
india_filters = ('New Delhi, India', 'Mumbai', 'Mumbai, India', 'New Delhi', 'India',
                 'Bengaluru, India')
australia_filters = ('Sydney, Australia', 'New South Wales', 'Melbourne, Australia', 'Sydney',
                     'Sydney, New South Wales', 'Melbourne, Victoria', 'Melbourne', 'Australia')
canada_filters = ('Toronto, Ontario', 'Toronto', 'Ontario, Canada', 'Toronto, Canada', 'Canada',
                  'Vancouver, British Columbia', 'Ontario', 'Victoria', 'British Columbia', 'Alberta',)
south_africa_filters = ('Johannesburg, South Africa', 'Cape Town, South Africa', 'South Africa')
nigeria_filters = ('Lagos, Nigeria')
kenya_filters = ('Nairobi, Kenya')
france_filters = ('Paris, France')
ireland_filters = ('Ireland')
new_zealand_filters = ('New Zealand')
pakistan_filters = ('Pakistan')
malaysia_filters = ('Malaysia')
uganda_filters = ('Kampala, Uganda', 'Uganda')
singapore_filters = ('Singapore')
germany_filters = ('Germany', 'Deutschland')
switz_filters = ('Switzerland')
uae_filters = ('United Arab Emirates', 'Dubai')
spain_filters = ('Spain')
belg_filters = ('Belgium')
phil_filters = ('Philippines')
hk_filters = ('Hong Kong')
ghana_filters = ('Ghana')

df['country_name'] = df['Location'].apply(lambda x: x.split(",")[-1].strip() if ("," in x) else x)
print(df)

# Changing strings found with filters into 3 digit codes
df['country_name'] = df['country_name'].apply(lambda x: 'USA' if x in us_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'GBR' if x in uk_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'IND' if x in india_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'AUS' if x in australia_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'CAN' if x in canada_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'ZAF' if x in south_africa_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'KEN' if x in kenya_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'NGA' if x in nigeria_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'SGP' if x in singapore_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'FRA' if x in france_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'NZL' if x in new_zealand_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'PAK' if x in pakistan_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'MYS' if x in malaysia_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'IRL' if x in ireland_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'UGA' if x in uganda_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'DEU' if x in germany_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'CHE' if x in switz_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'ARE' if x in uae_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'ESP' if x in spain_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'BEL' if x in belg_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'PHL' if x in phil_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'GHA' if x in ghana_filters else x)
df['country_name'] = df['country_name'].apply(lambda x: 'HKG' if x in hk_filters else x)

df['country_name'].value_counts()

vect = CountVectorizer(stop_words = 'english')
X_train_matrix = vect.fit_transform(df["newTweet"])

y = df["sentiment_target"]
X_train, X_test, y_train, y_test = train_test_split(X_train_matrix, y, test_size= 0.3)
model = MultinomialNB()
model.fit(X_train, y_train)
print (model.score(X_train, y_train))
print (model.score(X_test, y_test))
predicted_result = model.predict(X_test)
print("\n")
print(classification_report(y_test,predicted_result))

#Visualization 1 of sentiments
df = df[["Sentiment", "newTweet"]]
df.Sentiment.value_counts().plot.bar(figsize = (15,7))

# Create wordcloud
fig, (ax) = plt.subplots(1,1,figsize=[7, 5])
wc = WordCloud(width=600,height=400, background_color='White', colormap="Greens").generate(" ".join(df['newTweet']))

ax.imshow(wc,interpolation='bilinear')
ax.axis('off')
ax.set_title('Word cloud tweets');

# Visualization 2 of sentiment distribution
output_file("sentiment.html")
x = df.Sentiment.value_counts()

data = pd.Series(x).reset_index(name='value').rename(columns={'index':'sentiment'})
data['angle'] = data['value']/data['value'].sum() * 2*pi
data['color'] = Plasma[len(x)]

p = figure(plot_height=350, title="Sentiment", toolbar_location=None,
           tools="hover", tooltips="@sentiment: @value", x_range=(-0.5, 1.0))

p.wedge(x=0, y=1, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='sentiment', source=data)

p.axis.axis_label=None
p.axis.visible=False
p.grid.grid_line_color = None
show(p)

#Visualization 3 of country
data = dict(type='choropleth',
            colorscale = 'purp',
            locations = df_country['Country'],
            z = df_country['Tweets'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"Cantidad de tweets"}
            ) 

layout = dict(title = 'Cantidad de tweets por país',
              geo = dict(scope='world',
                         showlakes = False,
                         lakecolor = '#132630',
                         projection_type='natural earth')
             )

choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)

#tweets from users
#
#Aquí debe ir la manera de obtener el tweet de la caja de texto y guardarlo en un csv
#
#
