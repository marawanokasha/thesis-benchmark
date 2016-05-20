"""
Utility Functions
"""

from random import shuffle

import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from thesis.utils.WordUtils import stemtokenizer
from thesis.utils.patent import PatentIterable


def _preprocessor(document):
    """
    Preprocessor callback, called by the CountVectorizer to oversee the tokenization and preprocessing of words.
    Input Arguments:
        document: the entire document text
    """
    doc = " ".join(stemtokenizer(document))
    return doc


"""
Main Methods of the code
"""


# TF Bag of Words

def create_bag_of_words(patents, min_df=5):
    """
    Creates a bag of words representation from the raw article documents, then dumps both
    the bag or words and its vocabulary pickle files to GridFS
    Should only be called once to create the pickle files, afterwards, just read from the pickle files
    """

    print "Creating Standard TF Bag of Words"

    vectorizer = CountVectorizer(preprocessor=_preprocessor, min_df=min_df, stop_words='english')
    patents = PatentIterable(patents)
    try:
        # tf of the documents, consists of a sparse matrix
        bow = vectorizer.fit_transform(patents)
    except Exception as exc:
        print exc;

    print "Number of all Articles: " + str(len(patents))

    return bow, vectorizer.vocabulary_


def read_bag_of_words(field='content', testset=False):
    """
    Reads the Bag of words pickle representation back from GridFS and
    returns a tuple with the bow as the first element and the vocabulary
    elements as the 2nd element
    """

    bow_loaded = FileRepresentation.read_representation(
        get_label(GRAMS.UNI, field, FEATURE.BOW, DATA.VALUES, testset=testset))
    vocab_loaded = FileRepresentation.read_object(
        get_label(GRAMS.UNI, field, FEATURE.BOW, DATA.VOCABULARY, testset=testset))

    return (bow_loaded, vocab_loaded)


# TF-IDF Bag of Words

def create_tf_idf(bow):
    """ (SPARSE VERSION) Reads the bag of words representation from GridFS, then generates the TF-IDF representation """

    print "Creating TF-IDF Bag of Words"

    transformer = TfidfTransformer(norm=u'l2', use_idf=True)
    transformer.fit(bow)
    tf_idf = transformer.transform(bow)

    return tf_idf


def read_tf_idf(field='content', testset=False):
    tf_idf_loaded = FileRepresentation.read_representation(
        get_label(GRAMS.UNI, field, FEATURE.TFIDF, DATA.VALUES, testset=testset))
    return tf_idf_loaded


def _calculate_tf_idf():
    """
    DEPRECATED: Reads the bag of words representation from pickle, then generates the TF-IDF conversion
    """
    # Reading the bag of words from pickle files for creation of the tf-idf representation
    bag_of_words_and_vocab = read_bag_of_words()
    bag = bag_of_words_and_vocab[0]
    voc = bag_of_words_and_vocab[1]

    # TfidfTransformer takes a bag of words and computes its tf-idf counterpart
    transformer = TfidfTransformer(norm=u'l2')
    tf_idf = transformer.fit_transform(bag.toarray())

    print tf_idf

    # Getting non zero elements in the sparse matrix
    tf_idf_non_zero = scipy.sparse.find(tf_idf)
    # converting the non-zero tuple of lists elements to list of lists
    tf_idf_non_zero = [list(i) for i in tf_idf_non_zero]  # list of lists
    # transposing the list of lists to be able to sort it
    tf_idf_non_zero_transpose = [list(i) for i in zip(*tf_idf_non_zero)];

    def getKey(item):
        return item[2]

    # order the results descendingly according to the tf-idf value
    tf_idf_non_zero_sorted = sorted(tf_idf_non_zero_transpose, key=getKey, reverse=True)

    # Get first 100 words ordered by tf-idf value
    imp = tf_idf_non_zero_sorted[:100]

    print imp

    shuffle(imp)

    top_document_indices = [w[0] for w in imp]
    top_word_indices = [w[1] for w in imp]
    top_word_names = dict()

    for key, value in voc.iteritems():
        if value in top_word_indices:
            top_word_names[value] = key

    print top_document_indices
    print top_word_indices
    print top_word_names

    tf_idf_non_sparse = tf_idf.toarray()

    print len(tf_idf_non_sparse)

    for doc_index in top_document_indices:
        for word_index in top_word_indices:
            print "Document" + str(doc_index) + "," + str(top_word_names[word_index]) + "," + str(
                tf_idf_non_sparse[doc_index][word_index])


def sort_save_tfidf(field='content', testset=False):
    """ Reads tfidf from the database and sorts its columns by the avg tfidf (descendingly) and overwrite the matrix in the database.
    NOTE : If testset=True, sorting will be taken from the tfidf matrix/vocab of the Training set """
    if testset == True:  # Must sort according to Training set
        print 'Sorting Test set according to the sorting of the saved Training set matrix/vocab'
        train_vocab = FileRepresentation.read_object(
            get_label(GRAMS.UNI, field, FEATURE.BOW, DATA.VOCABULARY, testset=False))  # assuming train is sorted
        test_vocab = FileRepresentation.read_object(
            get_label(GRAMS.UNI, field, FEATURE.BOW, DATA.VOCABULARY, testset=True))  # the unsorted test vocab
        tfidf_test = FileRepresentation.read_representation(
            get_label(GRAMS.UNI, field, FEATURE.TFIDF, DATA.VALUES, testset=True))
        new_vocab = train_vocab
        old_vocab = test_vocab
        sorted_matrix = sort_matrix_by_vocab(tfidf_test, old_vocab, new_vocab, verbose=True)
    else:
        vocab = FileRepresentation.read_object(get_label(GRAMS.UNI, field, FEATURE.BOW, DATA.VOCABULARY, testset=False))
        tfidf = FileRepresentation.read_representation(
            get_label(GRAMS.UNI, field, FEATURE.TFIDF, DATA.VALUES, testset=False))
        sum_vector = tfidf.sum(0)  # sum over all rows...to produce a column vector (actually numpy.matrix)
        sum_vector = sum_vector.tolist()[0]  # python list
        vocab_inverted = {v: k for k, v in vocab.items()}  # inverse vocab

        print "Sorting tfidf for %s" % field
        # get list of sorted vocab by index
        sorted_vocab = []
        for i in range(len(vocab_inverted)):
            sorted_vocab.append(vocab_inverted[i])

        word_avgtfidf = zip(sorted_vocab, sum_vector)  # [ (word1,sumtfidf), (word2,sumtfidf)...etc]
        word_avgtfidf.sort(key=lambda x: x[1], reverse=True)

        # convert to dict with index as value
        print 'getting new vocab'
        new_vocab = {}
        for i in range(len(vocab_inverted)):
            key, _ = word_avgtfidf[i]
            new_vocab[key] = i
        print 'sorting'
        sorted_matrix = sort_matrix_by_vocab(tfidf, vocab, new_vocab, verbose=True)

    print 'saving to gridfs'
    # Overwrite in db
    FileRepresentation.save_object(new_vocab,
                                   get_label(GRAMS.UNI, field, FEATURE.BOW, DATA.VOCABULARY, testset=testset));
    FileRepresentation.save_object(sorted_matrix,
                                   get_label(GRAMS.UNI, field, FEATURE.TFIDF, DATA.VALUES, testset=testset));


def sort_matrix_by_avg_value(X, vocab):
    sum_vector = X.sum(0)  # sum over all rows...to produce a column vector (actually numpy.matrix)
    sum_vector = sum_vector.tolist()[0]  # python list
    vocab_inverted = {v: k for k, v in vocab.items()}  # inverse vocab

    print "Sorting tfidf for given matrix"
    # get list of sorted vocab by index
    sorted_vocab = []
    for i in range(len(vocab_inverted)):
        sorted_vocab.append(vocab_inverted[i])

    word_avgtfidf = zip(sorted_vocab, sum_vector)  # [ (word1,sumtfidf), (word2,sumtfidf)...etc]
    word_avgtfidf.sort(key=lambda x: x[1], reverse=True)

    # convert to dict with index as value
    print 'getting new vocab'
    new_vocab = {}
    for i in range(len(vocab_inverted)):
        key, _ = word_avgtfidf[i]
        new_vocab[key] = i
    print 'sorting'
    sorted_matrix = sort_matrix_by_vocab(X, vocab, new_vocab, verbose=True)
    return sorted_matrix


if __name__ == '__main__':
    # bow,voc = create_bag_of_words()
    # calculate_tf_idf()

    create_bag_of_words()
    create_tf_idf()
    sort_save_tfidf()
