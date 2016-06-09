import urllib2
import json
import math
import random
import time
import pprint
from thesis.utils.config import config, app_config

SAMPLE_RATIO = 0.03
MIN_NUMBER_OF_DOCS_TO_RETURN = 9
AGGREGATE_REQUEST_DATA = """
{
    "size": 0,
    "aggs": {
        "group_by_section": {
            "terms": {
                "field": "classification-ipc.section"
            },
            "aggs": {
                "group_by_class": {
                    "terms": {
                        "field": "classification-ipc.class"
                    },
                    "aggs": {
                        "group_by_subclass": {
                            "terms": {
                                "field": "classification-ipc.subclass"
                            }
                        }
                    }

                }
            }
        }
    }
}
"""

SECTION_CLASS_SUBCLASS_REQUEST = """
{
    "query": {
        "function_score": {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"classification-ipc.section": "{{section}}"} },
                        {"match": {"classification-ipc.class": "{{class}}"} }
                    ]
                }
            },
            "random_score": {
                "seed": "{{seed}}"
            }
        }
    },
    "size": {{size}}
}
"""
SEED = 123

VIABLE_CLASSES = {u'D-05': 26, u'D-04': 79, u'D-06': 113, u'D-01': 96, u'D-03': 69, u'D-02': 51, u'A-21': 13, u'A-23': 215, u'C-01': 476, u'C-02': 149, u'C-03': 138, u'C-04': 19, u'C-06': 11, u'C-07': 4197, u'C-08': 1292, u'C-09': 721, u'G-11': 943, u'G-10': 308, u'D': 463, u'H': 13422, u'B-60': 700, u'B-62': 225, u'B-65': 615, u'B-64': 19, u'B-66': 12, u'C-12': 2549, u'C-11': 47, u'C-10': 188, u'F-41': 96, u'H-01': 3943, u'H-03': 1098, u'H-02': 881, u'H-05': 1013, u'H-04': 6764, u'H-06': 17, u'A-01': 1862, u'C-25': 45, u'C-22': 16, u'C-23': 445, u'C-21': 11, u'E-06': 99, u'E-04': 297, u'E-05': 182, u'E-02': 98, u'E-03': 61, u'E-01': 116, u'C': 8965, u'G': 15452, u'B-82': 15, u'B-05': 688, u'G-21': 19, u'B-03': 15, u'B-01': 1186, u'C-30': 12, u'B-08': 153, u'A-61': 5176, u'A-62': 136, u'A-63': 400, u'F-16': 874, u'B-32': 628, u'B': 5412, u'F': 2843, u'F-03': 143, u'F-02': 510, u'F-01': 490, u'F-04': 187, u'B-29': 489, u'B-28': 19, u'B-25': 18, u'B-21': 30, u'B-23': 536, u'B-22': 20, u'D-21': 96, u'A-47': 504, u'A-45': 14, u'A-41': 12, u'A': 8103, u'E': 1076, u'E-21': 267, u'F-28': 57, u'F-21': 316, u'F-23': 113, u'F-25': 154, u'F-24': 53, u'B-41': 430, u'B-44': 19, u'G-08': 661, u'G-09': 785, u'G-04': 312, u'G-05': 780, u'G-06': 7781, u'G-07': 44, u'G-01': 2599, u'G-02': 1064, u'G-03': 996}

SIZE = 300

file_name = config.get("data", "sample_content_file")

training_doc_ids = []
with open(file_name, 'r') as file:
    i = 0
    for line in file:
        patent = json.loads(line)
        training_doc_ids.append(patent['id'])
        i+=1
print i
print "Finished loading training docs"

def get_no_of_sample_docs(total_docs):
    sample_docs = math.ceil(total_docs * SAMPLE_RATIO)
    return sample_docs if sample_docs > 0 else (MIN_NUMBER_OF_DOCS_TO_RETURN if total_docs > MIN_NUMBER_OF_DOCS_TO_RETURN else total_docs)

elasticsearch_url = config.get("data", "elastic_search_url")
data_str = urllib2.urlopen(elasticsearch_url, AGGREGATE_REQUEST_DATA).read()

data = json.loads(data_str)

total_number = data['hits']['total']

test_file = config.get("data", "sample_test_file")
validation_file = config.get("data", "sample_validation_file")

section_dict = {}

for section in VIABLE_CLASSES:
    section = section.split("-")[0]
    if section not in section_dict:
        viable_classes = filter(lambda clss: clss.split("-")[0] == section and len(clss.split("-")) > 1, VIABLE_CLASSES.keys())
        section_dict[section] = sorted(map(lambda x: x.split("-")[1], viable_classes))

print section_dict

test_count = 0
with open(test_file, 'w') as test_file_handle:
    for section in section_dict:
        print section_dict[section]
        classes = section_dict[section]
        for clss in classes:
            query = SECTION_CLASS_SUBCLASS_REQUEST\
                        .replace("{{section}}", section)\
                        .replace("{{class}}", clss)\
                        .replace("{{seed}}", str(SEED))\
                        .replace("{{size}}", str(SIZE)) \

            data_str = urllib2.urlopen(elasticsearch_url, query).read()

            data = json.loads(data_str)
            print "subclass request loaded for section %s, class %s" % (section, clss)

            for hit in data['hits']['hits']:
                patent = hit['_source']
                patent['id'] = hit['_id']
                if patent['id'] in training_doc_ids:
                    continue
                valid = False
                for classf in patent['classification-ipc']:
                    if classf['section'] == section and classf['class'] == clss:
                        valid = True
                if valid:
                    patent_str = json.dumps(patent)
                    test_file_handle.write(patent_str + "\n")
                    test_count += 1

                # i+=1
        # if i > 10: break

print "Test Count: " + str(test_count)

