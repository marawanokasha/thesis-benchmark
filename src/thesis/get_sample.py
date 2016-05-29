import urllib2
import json
import math
import pprint
from thesis.utils.config import config, app_config

SAMPLE_RATIO = 0.01
MIN_NUMBER_OF_DOCS_TO_RETURN = 3
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


ouptut_file = config.get("data", "sample_content_file")
i = 1
with open(ouptut_file, 'w') as file:
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

        for hit in data['hits']['hits']:
            patent = hit['_source']
            patent['id'] = hit['_id']
            patent_str = json.dumps(patent)
            file.write(patent_str + "\n")
        # i+=1
        # if i > 10: break


pprint.pprint(subclass_dict)
print len(subclass_dict)
