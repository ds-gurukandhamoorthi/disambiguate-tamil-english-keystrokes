import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

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

    # both = pd.concat([word_stat, eng_word_stat])

    both = pd.concat([word_stat[:20000], eng_word_stat[:20000]])

    both = (both
        .dropna(subset=['keystrokes']))

    keystrokes = both['keystrokes']
    labels = both['lang'] == 'tamil'

    pipe = make_pipeline(
            CountVectorizer(analyzer='char', ngram_range=(2,2), preprocessor=lambda s: f' {s} '),
            MultinomialNB()
            )

    pipe.fit(keystrokes, labels)
    print(f'Score on training set: {pipe.score(keystrokes,labels)}')

    pipe = make_pipeline(
            CountVectorizer(analyzer='char', ngram_range=(2,2), preprocessor=lambda s: f' {s} '),
            SVC(C=100)
            )
    pipe.fit(keystrokes, labels)
    print(f'Score on training set: {pipe.score(keystrokes,labels)}')
