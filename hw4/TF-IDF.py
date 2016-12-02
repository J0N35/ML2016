#!/usr/bin/python3
# coding: utf-8

import numpy as np

# reference: http://scikit-learn.org/stable/auto_examples/text/document_clustering.html

def vectorize(sequence):
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # initialize
    tfidf_vectorizer = TfidfVectorizer(min_df=2)
    # transform into tfidf vector
    title_vec = tfidf_vectorizer.fit_transform(sequence)
    
    return title_vec

def dim_reduction(vector):
    from sklearn.decomposition import TruncatedSVD # LSA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer
    from sklearn.cluster import KMeans
    
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(n_components=22, n_iter=20)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    lsa_vector = lsa.fit_transform(vector)
    
    return lsa_vector

def cluster(vector):
    from sklearn.cluster import KMeans

    class_nb = 20 # number of class/ cluster
        
    km = KMeans(n_clusters=class_nb, n_init=40, max_iter=500, tol=0.00001, n_jobs=-1)
    km.fit(vector)
    
    return km.labels_

def compare_export(test, label, output_path):    
    def compare(x):
        l1, l2 = x    
        return label[l1] == label[l2]
    
    result = [i for i in map(compare, test)]
    result_id = np.array(range(len(result)))
    answer = np.vstack((result_id, result)).T
    np.savetxt(output_path, answer, fmt='%i', delimiter=',', header='ID,Ans', comments='')

def preprocess(text):
    import nltk
    from nltk.corpus import stopwords
    from keras.preprocessing.text import text_to_word_sequence    
    
    # filter function
    def remove_stopword(x):
        # stem word
        x = stemmer.stem(x)
        # remove stop word and digit
        if 'model' in globals():
            if x not in model:
                return False
        if x in cached_stopwords:
            return False
        if x.isdigit():
            return False
        else:
            return True
    
    # split sentence and remove punctuation    
    title_sequence = [text_to_word_sequence(s) for s in text]
    
    # cache stopword
    cached_stopwords = nltk.corpus.stopwords.words('english')
    
    # load nltk's SnowballStemmer as variabled 'stemmer'
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")

    for idx, line in enumerate(title_sequence):
        title_sequence[idx] = [stemmer.stem(word) for word in filter(remove_stopword, line)]

    topic = []
    for line in title_sequence:
        topic.append(' '.join(line))
    
    return topic

def main(directory_path, output_path):
    import load_data

    directory_path = "dataset/"
    output_path = "output_TFIDF.csv"

    # ===== Import Data =====
    print('Importing Data...', end='')
    test_data, title_data, doc_data = load_data.import_data(directory_path)
    print('Success!')

    # ===== Proprocess Data =====
    title_sequence = preprocess(title_data)
    print('Stop words removed')

    # ===== TF-IDF =====
    tfidf_vector = vectorize(title_sequence)

    # ===== Dim reduction =====
    dr_vector = dim_reduction(tfidf_vector)

    # ===== Clustering =====
    print('Clustering with KMean...', end='')
    cluster_result = cluster(dr_vector)
    print('Finish')

    # ===== Compare with testing data and export result =====
    compare_export(test_data, cluster_result, output_path)
    print('Test Result Exported as {}'.format(output_path))

if __name__ == '__main__':
    from sys import argv
    
    # directory_path = "dataset/"
    # output_path = "output_TFIDF.csv"
    
    if (len(argv) < 2):
        directory_path = "dataset/"
        output_path = "output_TFIDF.csv"
    else:
        directory_path, output_path = argv[1], argv[2]
        
    main(directory_path, output_path)
    
    print('===== END =====')

