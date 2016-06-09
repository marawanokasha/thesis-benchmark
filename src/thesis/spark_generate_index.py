import json
import nltk
from nltk.tokenize import RegexpTokenizer
import string
import math
import time

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.classification import SVMWithSGD, SVMModel


STOP_WORDS = nltk.corpus.stopwords.words('english')
NUMBER_INDICATOR = "number_inidicator"
CURRENCY_INDICATOR = "currency_inidicator"
CHEMICAL_INDICATOR = "chemical_inidicator"
MIN_SIZE = 3
MIN_DOCUMENTS = 3

SVM_ITERATIONS = 1000
SVM_CONVERGENCE = 0.1
SVM_REG = 0.01

BM25_K = 1.5  # controls power of tf component
BM25_b = 0.75  # controls the BM25 length normalization

stemmer = nltk.stem.porter.PorterStemmer().stem

def stemtokenizer(text, doc_id):
    """ MAIN FUNCTION to get clean stems out of a text. A list of clean stems are returned """
    tokenizer = RegexpTokenizer(r'\s+', gaps=True)
    tokens = tokenizer.tokenize(text)
    stems = []  # result
    for token in tokens:
        stem = token
        stem = stem.strip(string.punctuation)
        if stem:
            if is_number(stem):
                stem = NUMBER_INDICATOR
            elif is_currency(stem):
                stem = CURRENCY_INDICATOR
            elif is_chemical(stem):
                stem = CHEMICAL_INDICATOR
            elif is_stopword(stem):
                stem = None
            else:
                stem = stemmer(token)
                stem = stem.strip(string.punctuation)
            if stem and len(stem) >= MIN_SIZE:
                stems.append((stem,{doc_id: 1}))
    return stems

def is_stopword(word):
  return word in STOP_WORDS

def is_number(str):
    """ Returns true if given string is a number (float or int)"""
    try:
        float(str.replace(",", ""))
        return True
    except ValueError:
        return False

def is_currency(str):
    return str[0] == "$"

def is_chemical(str):
    return str.count("-") > 3

def merge_postings(postings_list1, postings_list2):
    # key could be either a doc id or a term
    for key in postings_list2:
        if postings_list1.get(key):
            postings_list1[key] += postings_list2[key]
        else:
            postings_list1[key] = postings_list2[key]
    return postings_list1

def get_term_dictionary(terms):
    """
    Maps string terms to indexes in an array
    """
    term_dictionary = {}
    term_array = [None] * len(terms)
    def put(key):
        hashvalue = hashfunction(key, len(term_array))
        if term_array[hashvalue] == None:
            term_array[hashvalue] = key
            return hashvalue
        else:
            nextslot = rehash(hashvalue, len(term_array))
            while term_array[nextslot] != None:
                nextslot = rehash(nextslot, len(term_array))
            if term_array[nextslot] == None:
                term_array[nextslot] = key
                return nextslot
    def hashfunction(key, size):
        return hash(key) % size
    def rehash(oldhash, size):
        return (oldhash + 1) % size
    i = 0
    for term in terms:
        corresponding_index = put(term)
        term_dictionary[term] = corresponding_index
        i+=1
        if i%1000 == 0: print "finished " + str(i)
    return term_dictionary

def get_doc_index(term, postings_list, term_dictionary):
    #return [(doc_id, {term: postings_list[doc_id]}) for doc_id in postings_list]
    return [(doc_id, {term_dictionary[term]: postings_list[doc_id]}) for doc_id in postings_list]

def get_classes(ipc_classification):
    sections = []
    classes = []
    subclasses = []
    for classification in ipc_classification:
        # we do the check because some documents have repetitions
        section_name = classification['section']
        class_name = classification['section'] + "-" + classification['class']
        subclass_name = classification['section'] + "-" + classification['class'] + "-" + classification['subclass']
        if section_name not in sections:
            sections.append(section_name)
        if class_name not in classes:
            classes.append(class_name)
        if subclass_name not in subclasses:
            subclasses.append(subclass_name)
    return {"sections": sections, "classes": classes, "subclasses": subclasses}


def get_training_vector_old(classification, term_list, classifications, classification_key_name, number_of_terms):
    clss = 1 if classification in classifications[classification_key_name] else 0
    return LabeledPoint(clss, SparseVector(number_of_terms, term_list))

def get_training_vector(classification, term_list, classifications, number_of_terms):
    clss = 1 if classification in classifications else 0
    return LabeledPoint(clss, SparseVector(number_of_terms, term_list))


def calculate_tf_idf(tf, df, N):
    # laplace smoothing with +1 in case of term with no documents (useful during testing)
    return tf * math.log10((N+1) / (df + 1))


def calculate_bm25(tf, df, N, d_len, d_avg):
    idf = max(0, math.log10((N-df + 0.5)/(df+0.5))) # in rare cases where the df is over 50% of N, this could become -ve, so we guard against that
    tf_comp = float(((BM25_K + 1) * tf)) / ( BM25_K * ((1-BM25_b) + BM25_b*(float(d_len)/d_avg)) + tf)
    return tf_comp * idf


def calculate_rf(df_relevant, df_non_relevant):
    return math.log( (2 + (float(df_relevant)/max(1, df_non_relevant))), 2)


def calculate_tf_rf(tf, df_relevant, df_non_relevant):
    return tf * calculate_rf(df_relevant, df_non_relevant)


def compare_classifications(x,y):
    len_comp = cmp(len(x), len(y))
    if len_comp == 0:
        return cmp(x,y)
    return len_comp


def create_doc_index(term_index, term_dictionary):
    return term_index \
        .flatMap(lambda (term, postings_list): get_doc_index(term, postings_list, term_dictionary)) \
        .reduceByKey(lambda x, y: merge_postings(x, y))


def get_rf_stats(postings):
    a_plus_c = set(postings.keys())
    a_plus_b = set(classifications_index[classification])
    a = a_plus_c.intersection(a_plus_b)
    c = a_plus_c.difference(a_plus_b)
    size_a = len(a)
    size_c = len(c)
    return size_a, size_c


def get_rf_postings(postings):
    size_a, size_c = get_rf_stats(postings)
    return {docId: calculate_rf(size_a, size_c)
            for docId, tf in postings.items()}


def get_tf_rf_postings(postings):
    size_a, size_c = get_rf_stats(postings)
    return {docId: calculate_tf_rf(tf, size_a, size_c)
            for docId, tf in postings.items()}


def train_level_old(docs_with_classes, classification, classification_label):
    training_vectors = docs_with_classes.map(
        lambda (doc_id, (term_list, classifications)): get_training_vector_old(classification, term_list, classifications,
                                                                           classification_label, number_of_terms))
    svm = SVMWithSGD.train(training_vectors, iterations=SVM_ITERATIONS, convergenceTol=SVM_CONVERGENCE)
    return training_vectors, svm


def train_level(docs_with_classes, classification, number_of_terms):
    training_vectors = docs_with_classes.map(
        lambda (doc_id, (term_list, classifications)): get_training_vector(classification, term_list,
                                                                           classifications, number_of_terms))
    svm = SVMWithSGD.train(training_vectors, iterations=SVM_ITERATIONS, convergenceTol=SVM_CONVERGENCE, regParam=SVM_REG)
    return training_vectors, svm


def get_error(svm, test_vectors):
    labelsAndPreds = test_vectors.map(lambda p: (p.label, svm.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(test_vectors.count())
    return trainErr


def train_all(docs_with_classes):
    training_errors = {}
    for section in sections:
        training_vectors, svm = train_level(docs_with_classes, section, "sections")
        train_err = get_error(svm, training_vectors)
        training_errors[section] = train_err
    #
    with open(training_errors_output, 'w') as file:
        file.write(json.dumps(training_errors))
    #
    for clss in classes:
        training_vectors, svm = train_level(docs_with_classes, clss, "classes")
        train_err = get_error(svm, training_vectors)
        training_errors[clss] = train_err
    #
    with open(training_errors_output, 'w') as file:
        file.write(json.dumps(training_errors))
    #
    for subclass in subclasses:
        training_vectors, svm = train_level(docs_with_classes, subclass, "subclasses")
        train_err = get_error(svm, training_vectors)
        training_errors[subclass] = train_err
    #
    return training_errors


#sc = SparkContext("", "Generate Inverted Index Job")
save_parent_location = "hdfs://192.168.0.103/svm/"
file_name = "sample.json"
test_file_name = "sample.json"
#url = "/media/Work/workspace/thesis/benchmark/output/" + file_name
sample_location = save_parent_location + file_name
sample_test_location = save_parent_location + test_file_name
postings_list_output = save_parent_location + "postings_list_50000.json"
test_postings_list_output = save_parent_location + "test_postings_list_50000.json"
training_errors_output = save_parent_location + "training_errors.json"
model_output = save_parent_location + "models/" + "iter_" + str(SVM_ITERATIONS) + "_reg_" + str(SVM_REG) + "/"


# sc.addFile(sample_location)
# data = sc.textFile(SparkFiles.get("sample.json"))

data = sc.textFile(sample_location)

doc_count = data.count()

doc_objs = data.map(lambda x: json.loads(x))

doc_class_map = doc_objs.map(lambda x: (x['id'], get_classes(x['classification-ipc'])))
doc_classification_map = doc_class_map.map(lambda (doc_id, classification_obj): (doc_id, sorted(reduce(lambda x, lst: x + lst, classification_obj.values(), [])))).collectAsMap()

# contains [(classification,  list of docs)]
# second list comprehension is to get list of lists [["A", "B"],["A-01","B-03"]] to one list ["A", "B", "A-01","B-03"], we could have also used a reduce as in doc_classifications_map
classifications_index = doc_class_map.flatMap(lambda (doc_id, classifications_obj): [(classification, doc_id) for classification in [classif for cat in classifications_obj.values() for classif in cat]])\
    .groupByKey().map(lambda (classf, classf_docs): (classf, list(classf_docs))).collectAsMap()

sections = sorted(doc_class_map.flatMap(lambda (doc_id, classifications): classifications['sections']).distinct().collect())
classes = sorted(doc_class_map.flatMap(lambda (doc_id, classifications): classifications['classes']).distinct().collect())
subclasses = sorted(doc_class_map.flatMap(lambda (doc_id, classifications): classifications['subclasses']).distinct().collect())
classifications = sorted(classifications_index.keys(), cmp=compare_classifications)
# classifications = sorted(set(reduce(lambda x, lst: x + lst, map(lambda doc_id: classifications_index[doc_id], classifications_index), [])))

# Create Postings List
postings_lists = doc_objs.flatMap(lambda x: stemtokenizer(x['description'], x['id'])).reduceByKey(lambda x,y: merge_postings(x,y))
min_doc_postings_lists = postings_lists.filter(lambda (x,y): len(y) > MIN_DOCUMENTS)

# Load Postings Lists
# min_doc_postings_lists = sc.textFile(postings_list_output).map(lambda x: x.split(",", 1)).mapValues(lambda json_postings: json.loads(json_postings))

number_of_terms = min_doc_postings_lists.count()

# Save Postings List
# min_doc_postings_lists.map(lambda (term, postings_list): ",".join([term, json.dumps(postings_list)])).repartition(1).saveAsTextFile(postings_list_output)
min_doc_postings_lists.map(lambda postings: json.dumps(postings)).repartition(1).saveAsTextFile(postings_list_output)

all_terms = min_doc_postings_lists.keys().collect()

# gets a bit slower at the end but finishes
term_dictionary = get_term_dictionary(all_terms)

tf_postings = min_doc_postings_lists
tf_doc_index = create_doc_index(tf_postings, term_dictionary)

tf_idf_postings = tf_postings.mapValues(lambda postings: {docId:  calculate_tf_idf(tf, len(postings), doc_count) for docId, tf in postings.items()})
tf_id_doc_index = create_doc_index(tf_postings, term_dictionary)

# need to collect the document lengths since they are used in the BM25 calculation
doc_lengths_rdd = tf_doc_index.mapValues(lambda term_dictionary: reduce(lambda x, term: x + term_dictionary[term], term_dictionary, 0))
avg_doc_length = doc_lengths_rdd.map(lambda (term, count): count).reduce(lambda count1, count2: count1 + count2) / doc_count
doc_lengths_dict = doc_lengths_rdd.collectAsMap()

bm25_postings = tf_postings.mapValues(lambda postings: {docId: calculate_bm25(tf, len(postings), doc_count, doc_lengths_dict[docId], avg_doc_length) for docId, tf in postings.items()})
bm25_doc_index = create_doc_index(bm25_postings, term_dictionary)

training_errors = {}

VIABLE_CLASSES = {u'D-05': 26, u'D-04': 79, u'D-06': 113, u'D-01': 96, u'D-03': 69, u'D-02': 51, u'A-21': 13, u'A-23': 215, u'C-01': 476, u'C-02': 149, u'C-03': 138, u'C-04': 19, u'C-06': 11, u'C-07': 4197, u'C-08': 1292, u'C-09': 721, u'G-11': 943, u'G-10': 308, u'D': 463, u'H': 13422, u'B-60': 700, u'B-62': 225, u'B-65': 615, u'B-64': 19, u'B-66': 12, u'C-12': 2549, u'C-11': 47, u'C-10': 188, u'F-41': 96, u'H-01': 3943, u'H-03': 1098, u'H-02': 881, u'H-05': 1013, u'H-04': 6764, u'H-06': 17, u'A-01': 1862, u'C-25': 45, u'C-22': 16, u'C-23': 445, u'C-21': 11, u'E-06': 99, u'E-04': 297, u'E-05': 182, u'E-02': 98, u'E-03': 61, u'E-01': 116, u'C': 8965, u'G': 15452, u'B-82': 15, u'B-05': 688, u'G-21': 19, u'B-03': 15, u'B-01': 1186, u'C-30': 12, u'B-08': 153, u'A-61': 5176, u'A-62': 136, u'A-63': 400, u'F-16': 874, u'B-32': 628, u'B': 5412, u'F': 2843, u'F-03': 143, u'F-02': 510, u'F-01': 490, u'F-04': 187, u'B-29': 489, u'B-28': 19, u'B-25': 18, u'B-21': 30, u'B-23': 536, u'B-22': 20, u'D-21': 96, u'A-47': 504, u'A-45': 14, u'A-41': 12, u'A': 8103, u'E': 1076, u'E-21': 267, u'F-28': 57, u'F-21': 316, u'F-23': 113, u'F-25': 154, u'F-24': 53, u'B-41': 430, u'B-44': 19, u'G-08': 661, u'G-09': 785, u'G-04': 312, u'G-05': 780, u'G-06': 7781, u'G-07': 44, u'G-01': 2599, u'G-02': 1064, u'G-03': 996}

i=0
for classification in classifications:
    i+=1
    if i<= 84: continue
    if classification not in VIABLE_CLASSES.keys(): continue
    training_errors[classification] = {}
    representations_to_test = [("tf", tf_doc_index), ("tf-idf", tf_id_doc_index), ("bm25", bm25_doc_index)]
    #
    for name, doc_index in representations_to_test:
        docs_with_classes = doc_index.map(lambda (doc_id, terms): (doc_id, (terms, doc_classification_map[doc_id])))
        training_vectors, svm = train_level(docs_with_classes, classification, number_of_terms)
        svm.save(sc, model_output + name + "_" + classification + "_model.svm")
        # training_vectors = docs_with_classes.map(
        #     lambda (doc_id, (term_list, classifications)): get_training_vector(classification, term_list,
        #                                                                        classifications, number_of_terms))
        # svm = SVMWithSGD.train(training_vectors, iterations=SVM_ITERATIONS, convergenceTol=SVM_CONVERGENCE)
        train_err = get_error(svm, training_vectors)
        training_errors[classification][name] = train_err
    #
    # first intersection is to get (a), second difference is to get (c) (checkout tf-rf paper for reference)
    rf_postings = tf_postings.mapValues(get_rf_postings)
    rf_doc_index = create_doc_index(rf_postings, term_dictionary)
    docs_with_classes = rf_doc_index.map(lambda (doc_id, terms): (doc_id, (terms, doc_classification_map[doc_id])))
    training_vectors, svm = train_level(docs_with_classes, classification, number_of_terms)
    svm.save(sc, model_output + "rf_" + classification + "_model.svm")
    train_err = get_error(svm, training_vectors)
    training_errors[classification]["rf"] = train_err
    #
    # first intersection is to get (a), second difference is to get (c) (checkout tf-rf paper for reference)
    tf_rf_postings = tf_postings.mapValues(get_tf_rf_postings)
    tf_rf_doc_index = create_doc_index(tf_rf_postings, term_dictionary)
    docs_with_classes = tf_rf_doc_index.map(lambda (doc_id, terms): (doc_id, (terms, doc_classification_map[doc_id])))
    training_vectors, svm = train_level(docs_with_classes, classification, number_of_terms)
    svm.save(sc, model_output + "tf_rf_" + classification + "_model.svm")
    train_err = get_error(svm, training_vectors)
    training_errors[classification]["tf-rf"] = train_err







test_data = sc.textFile(sample_test_location)

test_doc_count = test_data.count()

test_doc_objs = test_data.map(lambda x: json.loads(x))

test_doc_class_map = test_doc_objs.map(lambda x: (x['id'], get_classes(x['classification-ipc'])))
test_doc_classification_map = test_doc_class_map.map(lambda (doc_id, classification_obj): (doc_id, sorted(reduce(lambda x, lst: x + lst, classification_obj.values(), [])))).collectAsMap()

# contains [(classification,  list of docs)]
# second list comprehension is to get list of lists [["A", "B"],["A-01","B-03"]] to one list ["A", "B", "A-01","B-03"], we could have also used a reduce as in doc_classifications_map
test_classifications_index = test_doc_class_map.flatMap(lambda (doc_id, classifications_obj): [(classification, doc_id) for classification in [classif for cat in classifications_obj.values() for classif in cat]])\
    .groupByKey().map(lambda (classf, classf_docs): (classf, list(classf_docs))).collectAsMap()

test_classifications = sorted(test_classifications_index.keys(), cmp=compare_classifications)

# Create Postings List
test_postings_lists = test_doc_objs.flatMap(lambda x: stemtokenizer(x['description'], x['id'])).reduceByKey(lambda x,y: merge_postings(x,y)).filter(lambda (term, postings): term in term_dictionary)

# Save Postings List
# min_doc_postings_lists.map(lambda (term, postings_list): ",".join([term, json.dumps(postings_list)])).repartition(1).saveAsTextFile(postings_list_output)
test_postings_lists.map(lambda postings: json.dumps(postings)).repartition(1).saveAsTextFile(test_postings_list_output)

test_tf_postings = test_postings_lists
test_tf_doc_index = create_doc_index(test_tf_postings, term_dictionary)

test_tf_idf_postings = tf_postings.mapValues(lambda postings: {docId:  calculate_tf_idf(tf, len(postings), doc_count) for docId, tf in postings.items()})
test_tf_id_doc_index = create_doc_index(test_tf_postings, term_dictionary)

# need to collect the document lengths since they are used in the BM25 calculation
test_doc_lengths_rdd = test_tf_doc_index.mapValues(lambda term_dictionary: reduce(lambda x, term: x + term_dictionary[term], term_dictionary, 0))
test_avg_doc_length = test_doc_lengths_rdd.map(lambda (term, count): count).reduce(lambda count1, count2: count1 + count2) / doc_count
test_doc_lengths_dict = test_doc_lengths_rdd.collectAsMap()

test_bm25_postings = test_tf_postings.mapValues(lambda postings: {docId: calculate_bm25(tf, len(postings), doc_count, test_doc_lengths_dict[docId], avg_doc_length) for docId, tf in postings.items()})
test_bm25_doc_index = create_doc_index(test_bm25_postings, term_dictionary)


class ContingencyTable:
    def __init__(self, tp, fp, fn, tn):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
        self.count = tp + fp + fn + tn


methods = ["tf", "tf-idf", "bm25", "rf", "tf_rf"]

contingency_tables = {"tf":{},"tf-idf":{},"bm25":{},"rf":{},"tf-rf":{}}

def get_contingency_table(binarySvm, doc_index, clssf):
    test_docs_with_classes = doc_index.map(
        lambda (doc_id, terms): (doc_id, (terms, test_doc_classification_map[doc_id])))
    test_vectors = test_docs_with_classes.map(
        lambda (doc_id, (term_list, classifications)): get_training_vector(clssf, term_list,
                                                                           classifications, number_of_terms))
    labelsAndPreds = test_vectors.map(lambda p: (p.label, binarySvm.predict(p.features)))
    tp = labelsAndPreds.filter(lambda (t, p): t == 1 and p == 1).count()
    fp = labelsAndPreds.filter(lambda (t, p): t == 0 and p == 1).count()
    fn = labelsAndPreds.filter(lambda (t, p): t == 1 and p == 0).count()
    tn = labelsAndPreds.filter(lambda (t, p): t == 0 and p == 0).count()
    ct = ContingencyTable(tp, fp, fn, tn)
    return ct


for section in sections:
    # binarySvm = SVMModel.load(sc, model_output + "tf" + "_" + section + "_model.svm")
    # if section != "A":
    #     ct = get_contingency_table(binarySvm, test_tf_doc_index, section)
    #     contingency_tables["tf"][section] = ct
    #     ct = get_contingency_table(binarySvm, test_tf_id_doc_index, section)
    #     contingency_tables["tf-idf"][section] = ct
    #     ct = get_contingency_table(binarySvm, test_bm25_doc_index, section)
    #     contingency_tables["bm25"][section] = ct

    test_rf_postings = test_tf_postings.mapValues(get_rf_postings)
    test_rf_doc_index = create_doc_index(test_rf_postings, term_dictionary)
    ct = get_contingency_table(binarySvm, test_rf_doc_index, section)
    contingency_tables["rf"][section] = ct

    test_tf_rf_postings = test_tf_postings.mapValues(get_tf_rf_postings)
    test_tf_rf_doc_index = create_doc_index(test_tf_rf_postings, term_dictionary)
    ct = get_contingency_table(binarySvm, test_tf_rf_doc_index, section)
    contingency_tables["tf-rf"][section] = ct
import cPickle as pickle
pickle.d
for clss in classes:
    binarySvm = SVMModel.load(sc, model_output + "tf" + "_" + clss + "_model.svm")

    ct = get_contingency_table(binarySvm, test_tf_doc_index, clss)
    contingency_tables["tf"][clss] = ct
    ct = get_contingency_table(binarySvm, test_tf_id_doc_index, clss)
    contingency_tables["tf-idf"][clss] = ct
    ct = get_contingency_table(binarySvm, test_bm25_doc_index, clss)
    contingency_tables["bm25"][clss] = ct

    test_rf_postings = test_tf_postings.mapValues(get_rf_postings)
    test_rf_doc_index = create_doc_index(test_rf_postings, term_dictionary)
    ct = get_contingency_table(binarySvm, test_rf_doc_index, clss)
    contingency_tables["rf"][clss] = ct

    test_tf_rf_postings = test_tf_postings.mapValues(get_tf_rf_postings)
    test_tf_rf_doc_index = create_doc_index(test_tf_rf_postings, term_dictionary)
    ct = get_contingency_table(binarySvm, test_tf_rf_doc_index, clss)
    contingency_tables["tf-rf"][clss] = ct


# for name, doc_index in representations_to_test:
#     docs_with_classes = doc_index.join(doc_class_map)
#     for section in sections:
#         training_vectors, svm = train_level(docs_with_classes, section, "sections")
#         train_err = get_error(svm, training_vectors)
#         training_errors[section] = train_err
#     #
#     with open(training_errors_output, 'w') as file:
#         file.write(json.dumps(training_errors))
#     #
#     for clss in classes:
#         training_vectors, svm = train_level(docs_with_classes, clss, "classes")
#         train_err = get_error(svm, training_vectors)
#         training_errors[clss] = train_err
#     #
#     with open(training_errors_output, 'w') as file:
#         file.write(json.dumps(training_errors))
#     #
#     for subclass in subclasses:
#         training_vectors, svm = train_level(docs_with_classes, subclass, "subclasses")
#         train_err = get_error(svm, training_vectors)
#         training_errors[subclass] = train_err
#     training_errors[name] = train_all(docs_with_classes)


with open(training_errors_output, 'w') as file:
    file.write(json.dumps(training_errors))


doc_index.take(100)

term_dictionary = sc.parallelize(term_dictionary)


postings_lists.take(100)
