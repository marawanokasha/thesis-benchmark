import json
from utils.config import config


file_name = config.get("data", "sample_content_file")

with open(file_name, 'r') as file:
    i = 0
    for line in file:
        patent = json.loads(line)
        print patent['id']
        i+=1
print i