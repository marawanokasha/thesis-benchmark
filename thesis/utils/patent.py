from time import time

class Patent:

    def __init__(self, dict):
        self.id = dict['_id']
        self.type = dict['_type']
        self.index = dict['_index']
        self.docNumber = dict['_source']['patent-doc-number']
        self.classification = dict['_source']['classification-ipc']
        self.classes = self._get_classes()
        self.title = dict['_source']['invention-title']
        self.description = dict['_source']['description']

    def _get_classes(self):
        return list(set(["".join(classification.values()) for classification in self.classification]))



class PatentIterable:
    """
    Iterable that iterates over the patent collection
    """
    def __init__(self, patents):
        self.collection = patents
        self.current = 0
        self.count = len(self.collection)
        self.start_time = time()

    def __iter__(self):
        """ Returns an instance of the iterable """
        return self

    def __len__(self):
        """ Returns length of the collection """
        return self.count

    def next(self):
        """
        Called with every iteration of a for loop
        """
        if self.current >= self.count or self.current > 30:
            raise StopIteration

        if self.current % 1000 == 0 and self.current != 0:
            print "Finished iterating over " + str(self.current) + " articles"

            print "Time elapsed for importing 1000 articles: " + str(time() - self.start_time)
            self.start_time = time()

        content = self.collection[self.current].title  + self.collection[self.current].description

        self.current += 1

        return content


def get_patents(dicts):
    patents = []
    for dict in dicts:
        patent = Patent(dict)
        patents.append(patent)

    return patents