import os
import sys
import json
import csv
import numpy as np
import emoji
import urllib
import pprint
import logging
import requests
import psycopg2
import pandas as pd
import nltk
import matplotlib.pyplot as plt

nltk.download('stopwords')
from searchtweets import ResultStream, gen_request_parameters, load_credentials
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from json import JSONDecodeError
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score

import urllib3
from requests.exceptions import RequestException

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def pretty_print(data):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(data)


def get_auth_token(logger, request_session, url: str, headers: dict = None) -> dict:
    """
    todo: Ensure the tweets are hash based
    todo: make the creds configurable
    :param logger:
    :param request_session:
    :param url:
    :param headers:
    :return:
    """
    resp = None
    try:
        request_session.auth = ("g24XLVQXHp4W6yMlzARk5Canb", "kjCnCV6FqasA0NLFxWLTBxzFsIunJlcPxLNHdzSegJwxTKIlS0")
        resp = request_session.post(url, headers=headers, verify=False)
        if not resp.status_code == 200:
            sys.exit()
        data = json.loads(resp.text)
    except (RequestException, JSONDecodeError) as err:
        logger.error(err)
        sys.exit()
    return data


def get_tweets(logger, request_session, url: str, headers: dict = None) -> dict:
    """
    todo: Ensure the tweets are hash based
    :param logger:
    :param request_session:
    :param url:
    :param headers:
    :return:
    """
    resp = None
    try:
        request_session.auth = None
        resp = request_session.get(url, headers=headers, verify=False)
        if not resp.status_code == 200:
            logger.error(resp.text)
            sys.exit()
        data = json.loads(resp.text)
    except (RequestException, JSONDecodeError) as err:
        logger.error(err)
        sys.exit()
    return data


def get_tweets_stream(query_search, max=500):
    combined_tweets = {'data': []}
    search_args = load_credentials(filename="./search_tweets_creds.yaml",
                                   yaml_key="search_tweets_v2",
                                   env_overwrite=False)
    query = gen_request_parameters(query_search, results_per_call=100)
    rs = ResultStream(
        request_parameters=query,
        max_tweets=max,
        max_requests=500,
        **search_args)
    paginated_tweets = list(rs.stream())
    for pagination in paginated_tweets:
        for tweet in pagination['data']:
            combined_tweets['data'].append(tweet)
    return combined_tweets


def textblob_polarity(text):
    blob_txt = TextBlob(text)
    polarity = blob_txt.sentiment.polarity
    return polarity


def vader_polarity(text):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(text)
    return sentiment_dict['compound']


def confidence_and_sentiment(vader_polarity: int, blob_polarity: int) -> tuple:
    """
    We convert polarities over .6 threshold to positive
    We convert polarities under 0 threshhold to negative

    subsequently when we clean the data we will keep anything with confidence over .65
    then convert anything positive to 0
    and anything negative to 1

    :param vader_polarity:
    :param blob_polarity:
    :return:
    """
    sentiment = 'neutral'
    avg_polarity = (vader_polarity + blob_polarity) / 2
    polarity_range = vader_polarity - blob_polarity
    if vader_polarity > .5:
        sentiment = 'positive'
    if vader_polarity < -0.05:
        sentiment = 'negative'
    confidence = 1 if polarity_range < .5 else 0
    return confidence, sentiment


def get_word_list(path):
    word_list = []
    with open(path) as f:
        for line in f:
            word_list.append(line.strip())
    return word_list

class Postgresql:
    def __init__(self, commit=False):
        self.conn = psycopg2.connect("dbname=CandidateAnalysis user=postgres password=200204178 host=127.0.0.1")
        self.conn.autocommit = commit

    def get_candidates(self):
        cur = self.conn.cursor()
        cur.execute('Select * from candidates')
        records = cur.fetchall()
        return records

    def insert_candidate_tweet(self, tweet, candidate_id):
        cur = self.conn.cursor()
        query = "INSERT INTO tweets(tweet, candidate) VALUES (%s, %s) ON CONFLICT DO NOTHING;; Select * from tweets where tweet = %s"
        cur.execute(query, (tweet, candidate_id, tweet))
        records = cur.fetchall()
        return records[0][0]

    def insert_sentiment(self, candidate_id, tweet_id, polarity, confidence, sentiment):
        cur = self.conn.cursor()
        query = "INSERT INTO sentiments(candidate_id, tweet_id, polarity, confidence, sentiment) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING;"
        cur.execute(query, (candidate_id, tweet_id, polarity, confidence, sentiment))
        return query


    def insert_svm_results(self, acc, auc, f1, svm_precision, recall, score, candidate_id):
        cur = self.conn.cursor()
        query = "INSERT INTO svm(acc, auc, f1, svm_precision, recall, score, candidate) VALUES (%s,%s,%s,%s,%s, %s, %s)"
        cur.execute(query, (acc, auc, f1, svm_precision, recall, score, candidate_id))
        return query


class CSV:
    filename = None
    fp = None
    writer = None

    def __init__(self, filename):
        self.filename = filename
        self.fp = open(self.filename, 'w', encoding='utf8')
        self.writer = csv.writer(self.fp)

    def close(self):
        self.fp.close()

    def write(self, elems):
        self.writer.writerow(elems)

    def size(self):
        return os.path.getsize(self.filename)

    def fname(self):
        return self.filename


class SVMModel:
    def __init__(self, path):
        self.data = pd.read_csv(path)

    def copy_of_data(self):
        return self.data.copy()

    @staticmethod
    def clean_data(pd_data):
        data_clean = pd_data[pd_data['confidence'] > 0.65]
        data_clean['sentiment'] = data_clean['sentiment']. \
            apply(lambda x: 1 if x == 'negative' else 0)

        # data_clean['text_clean'] = data_clean['text'].apply(lambda x: BeautifulSoup(x, "lxml").text)
        data_clean = data_clean.loc[:, ['text', 'sentiment']]
        return data_clean

    @staticmethod
    def test_and_train(cleaned_data):
        train, test = train_test_split(cleaned_data, test_size=0.2, random_state=1)
        x_train = train['text'].values
        x_test = test['text'].values
        y_train = train['sentiment']
        y_test = test['sentiment']
        return x_train, x_test, y_train, y_test

    @staticmethod
    def vectorizing():
        def tokenize(text):
            tknzr = TweetTokenizer()
            return tknzr.tokenize(text)

        # def stem(doc):
        #    return (stemmer.stem(w) for w in analyzer(doc))

        en_stopwords = set(stopwords.words("english"))

        vectorizer = CountVectorizer(
            analyzer='word',
            tokenizer=tokenize,
            lowercase=True,
            ngram_range=(1, 1),
            stop_words=en_stopwords)
        return vectorizer

    @staticmethod
    def grid_svm(x_train, x_test, y_train, y_test, vectorizer):
        kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        np.random.seed(1)

        pipeline_svm = make_pipeline(vectorizer,
                                     SVC(probability=True, kernel="linear", class_weight="balanced"))

        grid_svm = GridSearchCV(pipeline_svm,
                                param_grid={'svc__C': [0.01, 0.1, 1]},
                                cv=kfolds,
                                scoring="roc_auc",
                                verbose=1,
                                n_jobs=-1)

        grid_svm.fit(x_train, y_train)
        grid_svm.score(x_test, y_test)
        return grid_svm

    @staticmethod
    def grid_svm_results(model, x, y):
        pred_proba = model.predict_proba(x)[:, 1]
        pred = model.predict(x)

        auc = roc_auc_score(y, pred_proba)
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred)
        prec = precision_score(y, pred)
        rec = recall_score(y, pred)
        result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
        return result

    @staticmethod
    def get_roc_curve(model, x, y):
        pred_proba = model.predict_proba(x)[:, 1]
        fpr, tpr, _ = roc_curve(y, pred_proba)
        return fpr, tpr

    @staticmethod
    def plot_learning_curve(grid_svm, x_train, y_train, title='', ylim=None, figsize=(14, 8)):
        train_sizes, train_scores, test_scores = \
            learning_curve(grid_svm.best_estimator_, x_train, y_train, cv=5, n_jobs=-1,
                           scoring="roc_auc", train_sizes=np.linspace(.1, 1.0, 10), random_state=1)
        plt.figure(figsize=figsize)
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="lower right")
        return plt


def keep_only_sentiments(text, combined_list):
    # if 'panther' not in text.lower():
    #     return False
    for x in combined_list:
        if x.lower() in text.lower().split(' '):
            return True
    return False


class Tweets:
    def __init__(self, logger, request_session):
        self.logger = logger
        self.request_session = request_session

    def test_model_new_csv(self, grid_svm, sentiment_list):
        csv_path = './tweets_predicted_via_model_black_adam.csv'

        # streaming twitter data
        tweets = get_tweets_stream('#blackadam lang:en -is:retweet', max=10000)

        # processing tweets
        csv_cls = CSV(csv_path)
        csv_cls.write(['sentiment', 'text'])
        for tweet in tweets['data']:
            single_line_text = tweet['text'].replace('\n', ' ')
            if keep_only_sentiments(single_line_text, sentiment_list):
                text = emoji.demojize(single_line_text)
                prediction = grid_svm.predict([text])[0]
                sentiment = 'positive' if prediction == 0 else 'negative'
                csv_cls.write([sentiment, text])
                # pretty_print(tweet['text']) --> Only using this for debugging purposes
        csv_cls.close()

    def get_reviews(self):
        """
        todo: fix the query_params
        :return:
        """
        csv_path = './candidate_tweets.csv'

        # call postgres candidates table
        postgres_cls = Postgresql(True)
        candidates = postgres_cls.get_candidates()
        # streaming twitter data
        for candidate in candidates:
            candidate_postgres_id = candidate[0]
            candidate_name = candidate[1].lower()
            candidate_party = candidate[2].lower()
            csv_path = f'./candidate_tweets_{candidate_name.split(" ")[0]}.csv'
            other_candidate = 'herschel' if candidate_name == 'raphael warnock' else 'warnock'
            tweets = get_tweets_stream(f'"{candidate_party} {candidate_name}" -{other_candidate} is lang:en -is:retweet', max=100)

            # processing tweets
            positive_word_list = get_word_list('positive-words.txt')
            negative_word_list = get_word_list('negative-words.txt')
            combined_list = positive_word_list + negative_word_list
            csv_cls = CSV(csv_path)
            csv_cls.write(['Text Blob Polarity', 'Vader Polarity', 'confidence', 'sentiment', 'text'])
            for tweet in tweets['data']:
                single_line_text = tweet['text'].replace('\n', ' ')
                if keep_only_sentiments(single_line_text, combined_list):
                    text = emoji.demojize(single_line_text)
                    txt_blob_polarity = textblob_polarity(text)
                    txt_vader_polarity = vader_polarity(text)
                    confidence, sentiment = confidence_and_sentiment(txt_vader_polarity, txt_blob_polarity)
                    csv_cls.write([txt_blob_polarity, txt_vader_polarity, confidence, sentiment, text])
                    # tweet_postgres_id = postgres_cls.insert_candidate_tweet(text, candidate_postgres_id)
                    # postgres_cls.insert_sentiment(candidate_postgres_id, tweet_postgres_id, txt_vader_polarity, confidence, sentiment)
            csv_cls.close()
            # Lets start training our model
            svm_cls = SVMModel(csv_path)
            svm_pd = svm_cls.copy_of_data()
            clean_pd = svm_cls.clean_data(svm_pd)
            x_train, x_test, y_train, y_test = svm_cls.test_and_train(clean_pd)
            vectorizer = svm_cls.vectorizing()
            grid_svm = svm_cls.grid_svm(x_train, x_test, y_train, y_test, vectorizer)
            pretty_print(grid_svm.cv_results_)
            print(f'best score: {grid_svm.best_score_} \n')
            print('-------RESULTS-------------')
            svm_results = svm_cls.grid_svm_results(grid_svm.best_estimator_, x_test, y_test)
            # postgres_cls.insert_svm_results(svm_results['acc'], svm_results['auc'], svm_results['f1'], svm_results['precision'], svm_results['recall'], grid_svm.best_score_, candidate_postgres_id)
            pretty_print(svm_results)
            roc_svm = svm_cls.get_roc_curve(grid_svm.best_estimator_, x_test, y_test)
            # plotting vals for true positive
            fpr, tpr = roc_svm
            plt.figure(figsize=(14, 8))
            plt.plot(fpr, tpr, color="red")
            plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Roc curve')
            plt.show()

            # plotting learning curve
            lrn_plt = svm_cls.plot_learning_curve(grid_svm, x_train, y_train)
            lrn_plt.show()
            # self.test_model_new_csv(grid_svm, combined_list)


session = requests.session()
nlp_logger = logging.getLogger(__name__)
tweet_cls = Tweets(nlp_logger, request_session=session)
tweet_cls.get_reviews()
