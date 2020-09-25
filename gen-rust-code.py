import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    word_stat = pd.read_csv('data/tamil-word-stats.csv')
    mapping = pd.read_csv('data/tamil_inscript_mapping.csv')

    word_stat.set_index('word')
    mapping.set_index('tamil_word')

    # We can only process when we have valid inscript keystrokes.
    word_stat = (word_stat
            .join(mapping)
            .dropna(subset=['inscript_keystrokes']))

    word_stat['lang'] = 'tamil'


    eng_word_stat = pd.read_csv('data/english.csv', header=None, delimiter='\t', names=['word', 'count'])
    eng_word_stat['lang'] = 'eng' 
    eng_word_stat['keystrokes'] = eng_word_stat['word']

    word_stat['keystrokes'] = word_stat['inscript_keystrokes']
    word_stat = word_stat[['word', 'count', 'keystrokes','lang']]

    both_full = pd.concat([word_stat, eng_word_stat])

    rs = ShuffleSplit(n_splits=1, test_size=.02, random_state=0)
    _, smaller_indices = next(rs.split(both_full))
    both = both_full.iloc[smaller_indices]
    # both = pd.concat([word_stat[:20000], eng_word_stat[:20000]])

    both = (both
        .dropna(subset=['keystrokes']))

    keystrokes = both['keystrokes']
    labels = both['lang'] == 'tamil'

    pipe = make_pipeline(
            CountVectorizer(analyzer='char', ngram_range=(1,2), preprocessor=lambda s: f' {s} '),
            LogisticRegression()
            )
    pipe.fit(keystrokes, labels)
    print(f'Score on training set: {pipe.score(keystrokes,labels)}')

    unigrams_and_bigrams = pipe.named_steps['countvectorizer'].get_feature_names()
    lm_coefs = pipe.named_steps['logisticregression'].coef_
    for ngrm, coef in zip(unigrams_and_bigrams, lm_coefs[0]):
        print(f'lm_coefs.insert(String::from("{ngrm}"),{coef});')

    # pipe = make_pipeline(
    #         CountVectorizer(analyzer='char', ngram_range=(1,2), preprocessor=lambda s: f' {s} ', binary=True),
    #         RidgeClassifier(alpha=0.1)
    #         )
    # pipe.fit(keystrokes, labels)
    # print(f'Score on training set: {pipe.score(keystrokes,labels)}')
