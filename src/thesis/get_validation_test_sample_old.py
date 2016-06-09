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
                        {"match": {"classification-ipc.class": "{{class}}"} },
                        {"match": {"classification-ipc.subclass": "{{subclass}}"} }
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

section_data = data['aggregations']['group_by_section']['buckets']

sections = map(lambda x: x['key'], section_data)

subclass_dict = {}

for section in section_data:
    class_data = section['group_by_class']['buckets']
    for clss in class_data:
        subclass_data = clss['group_by_subclass']['buckets']
        for subclass in subclass_data:
            subclass_path = section['key'] + clss['key'] + subclass['key']
            subclass_dict[subclass_path] = {
                "number": get_no_of_sample_docs(subclass['doc_count']),
                "path": {
                    "section": section['key'],
                    "class": clss['key'],
                    "subclass": subclass['key']
                }
            }

print "Number of subclasses: " + str(len(subclass_dict))

test_file = config.get("data", "sample_test_file")
validation_file = config.get("data", "sample_validation_file")

validation_count = 0
test_count = 0
with open(test_file, 'w') as test_file_handle, open(validation_file, 'w') as validation_file_handle:
    for key in subclass_dict:
        print subclass_dict[key]
        query = SECTION_CLASS_SUBCLASS_REQUEST\
                    .replace("{{section}}", subclass_dict[key]['path']['section'])\
                    .replace("{{class}}", subclass_dict[key]['path']['class'])\
                    .replace("{{subclass}}", subclass_dict[key]['path']['subclass'])\
                    .replace("{{seed}}", str(SEED))\
                    .replace("{{size}}", str(subclass_dict[key]['number'])) \

        data_str = urllib2.urlopen(elasticsearch_url, query).read()

        data = json.loads(data_str)
        print "subclass request loaded"

        for hit in data['hits']['hits']:
            patent = hit['_source']
            patent['id'] = hit['_id']
            if patent['id'] in training_doc_ids:
                continue
            patent_str = json.dumps(patent)
            random.seed(time.time())
            if random.randint(0, 1) == 0:
                test_file_handle.write(patent_str + "\n")
                test_count += 1
            else:
                validation_file_handle.write(patent_str + "\n")
                validation_count += 1

                # i+=1
        # if i > 10: break


print "Test Count: " + str(test_count)
print "Validation Count: " + str(validation_count)

pprint.pprint(subclass_dict)
print len(subclass_dict)
