import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

if __name__ == '__main__':

    winedf = pd.read_csv('../input/winemag-data_first150k.csv')
    winedf.head()

    #check to see if there's any missing data, datatypes seem reasonable, etc.
    winedf.info()

    #check to see if there are any duplicates in description column - remove duplicates
    winedf[winedf['description'].duplicated(keep = False)].sort_values('description').head(8)
    winedf = winedf.drop_duplicates('description')

    #check to see how many wine varieties there are
    #limit to only 1500 to narrow down to 15 types
    variety_df = winedf.groupby('variety').filter(lambda x: len(x) > 1500)
    varieties = variety_df['variety'].value_counts().index.tolist()
    fig, ax = plt.subplots(figsize = (25, 10))
    sns.countplot(x = variety_df['variety'], order = varieties, ax = ax)
    plt.xticks(rotation = 90)
    plt.show()

    #print(variety_df['variety'].unique())

    #add punctuation to stop_words and union it with sklearn's default English stop_words list
    #vectorize descriptions
    punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
    stop_words = text.ENGLISH_STOP_WORDS.union(punc)
    desc = variety_df['description'].values
    vectorizer = TfidfVectorizer(stop_words = stop_words)
    X = vectorizer.fit_transform(desc)

    #look at subset of the words that are included in the vectorizing
    word_features = vectorizer.get_feature_names()
    word_features[550:575]

    #need to override default sklearn tokenizer to include stemming
    #also will use regular expression to only include words and no numbers
    stemmer = SnowballStemmer('english')
    tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

    def tokenize(text):
        '''
        function to pass through as argument in TfidfVectorizer
        transforms text to lowercase, tokenizes based off regular expression, and stems the words
        '''
        return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]

    #new vectorizer with tokenize function, look at subset to see how words were stemmed
    vectorizer2 = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize)
    X2 = vectorizer2.fit_transform(desc)
    word_features2 = vectorizer2.get_feature_names()
    word_features2[:50]

    #add max_features parameter to limit feature space
    vectorizer3 = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize, max_features = 1000)
    X3 = vectorizer3.fit_transform(desc)
    words = vectorizer3.get_feature_names()

    #fit kMeans - lower n_init to 5 instead of default 10 to quicken runtime
    #n_init is the number of times algorithm runs kMeans with different initial centroids
    kmeans = KMeans(n_clusters = 15, n_init = 5, n_jobs = -1)
    kmeans.fit(X3)

    #print out top 10 common words for each cluster
    common_words = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
    for num, centroid in enumerate(common_words):
        print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))

    #add cluster assignment to dataframe
    variety_df['cluster'] = kmeans.labels_

    #heatmap to show count of varieties in each cluster
    clusters = variety_df.groupby(['cluster', 'variety']).size()
    fig2, ax2 = plt.subplots(figsize = (30, 15))
    sns.heatmap(clusters.unstack(level = 'variety'), ax = ax2, cmap = 'Reds')
    ax2.set_xlabel('variety', fontdict = {'weight': 'bold', 'size': 24})
    ax2.set_ylabel('cluster', fontdict = {'weight': 'bold', 'size': 24})
    for label in ax2.get_xticklabels():
        label.set_size(16)
        label.set_weight("bold")
    for label in ax2.get_yticklabels():
        label.set_size(16)
        label.set_weight("bold")
    plt.show()
