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
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy

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
            CountVectorizer(analyzer='char', ngram_range=(2,2), preprocessor=lambda s: f' {s} '),
            MultinomialNB()
            )

    pipe.fit(keystrokes, labels)
    print(f'Score on training set: {pipe.score(keystrokes,labels)}')

    pipe = make_pipeline(
            CountVectorizer(analyzer='char', ngram_range=(2,2), preprocessor=lambda s: f' {s} '),
            LinearSVC(C=0.1)
            )
    pipe.fit(keystrokes, labels)
    print(f'Score on training set: {pipe.score(keystrokes,labels)}')

    pipe = make_pipeline(
            CountVectorizer(analyzer='char', ngram_range=(1,2), preprocessor=lambda s: f' {s} '),
            LogisticRegression()
            )
    pipe.fit(keystrokes, labels)
    print(f'Score on training set: {pipe.score(keystrokes,labels)}')

    pipe = make_pipeline(
            CountVectorizer(analyzer='char', ngram_range=(1,2), preprocessor=lambda s: f' {s} ', binary=True),
            RidgeClassifier(alpha=0.1)
            )
    pipe.fit(keystrokes, labels)
    print(f'Score on training set: {pipe.score(keystrokes,labels)}')

    pipe = make_pipeline(
            CountVectorizer(analyzer='char', ngram_range=(1,2), preprocessor=lambda s: f' {s} '),
            SGDClassifier(penalty='elasticnet', loss='log', alpha=0.000001)
            )
    pipe.fit(keystrokes, labels)
    print(f'Score on training set: {pipe.score(keystrokes,labels)}')

    from sklearn.manifold import Isomap
    iso = Isomap(n_components=2)


    vect = CountVectorizer(analyzer='char', ngram_range=(2,2))
    vect.fit(both['keystrokes'])
    X = vect.transform(both['keystrokes'])

    y = both['lang'] == 'tamil'

    iso = Isomap(n_components=2)
    projection = iso.fit_transform(X)

    plt.scatter(projection[:,0], projection[:,1], c=y, cmap='jet', alpha=0.1)
    plt.colorbar(ticks=[False, True])

    plt.show()

    predicted = cross_val_predict(pipe, keystrokes, labels)

    confusion_matrix(labels, predicted)

    print(precision_score(labels, predicted))
    print(recall_score(labels, predicted))
    print(f1_score(labels, predicted))

    print(roc_auc_score(labels, predicted))

    pipe = make_pipeline(
            CountVectorizer(analyzer='char', ngram_range=(2,2), preprocessor=lambda s: f' {s} '),
            DecisionTreeClassifier()
            )

    pipe = make_pipeline(
            CountVectorizer(analyzer='char', ngram_range=(1,3), preprocessor=lambda s: f' {s} '),
            RandomForestClassifier(n_estimators=2000, n_jobs=-1)
            )
    pipe.fit(keystrokes, labels)
    print(f'Score on training set: {pipe.score(keystrokes,labels)}')

    features_dict = {k: v for (k,v) in zip(pipe.named_steps['countvectorizer'].get_feature_names(), pipe.named_steps['randomforestclassifier'].feature_importances_)}
    imp_features = sorted(features_dict.items(), key=lambda x:x[1])
    for i in imp_features:
        print(i)

    for k, v in features_dict.items():
        if v == 0:
            print (k, v)

    entropy([v for _, v in features_dict.items() ])
