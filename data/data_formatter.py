import json
import mmcv

import os.path

with open('./train/bepro.json') as json_file:
    data = json.load(json_file)

for match_name,img_list in data["train"].items():
    print (match_name)
    with open(img_list) as f:
        for path in f:
            img_path = os.path.join(data["root"], path.strip())
            img = mmcv.imread(img_path)
            print (img.shape)
