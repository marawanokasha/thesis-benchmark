import json
from thesis.utils.config import config


file_name = config.get("data", "sample_content_file")

with open(file_name, 'r') as file:
    for line in file:
        patent = json.loads(line)
        print patent['id']