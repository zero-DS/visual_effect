import json
import os
#{"file_name": "felt_person_001.png", "text": "s3wnf3lt man"}
dirname = "test_dataset"
paths = os.listdir(dirname)
diclist = []
for p in paths:
    keyword = p.split('_')[1]
    txt = "s3wnf3lt " + keyword
    dic = {'file_name' : p, 'text': txt}
    diclist.append(dic)
with open('metadata.jsonl', 'w') as f:
    for dic in diclist:
        json_line = json.dumps(dic)
        f.write(json_line + '\n')

