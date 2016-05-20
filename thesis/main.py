import json
from thesis.utils.config import config, app_config
from thesis.utils.patent import get_patents, PatentIterable
from thesis.preprocessing.tf_idf import create_bag_of_words, create_tf_idf
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

def main():
    file_name = config.get("data", "data_file")
    data = json.load(open(file_name))

    data = data['hits']['hits']

    patents = get_patents(data)

    all_classes = set()
    for p in patents:
        all_classes.update(p.classes)

    one_hot_representations = {}

    for i, c in enumerate(all_classes):
        bits = [0] * len(all_classes)
        bits[i] = 1
        one_hot_representations[c] = bits

    bow, vocabulary = create_bag_of_words(patents, min_df=0.005)

    tf_idf = create_tf_idf(bow)

    print len(patents)

    X = bow
    Y = []
    for p in patents:
        y_classes = []
        for c in p.classes:
            y_classes.append(one_hot_representations[c])
        Y.append(y_classes)

    classif = OneVsRestClassifier(SVC(kernel='linear'))

    classif.fit(X, Y)


if __name__ == '__main__':
    main()