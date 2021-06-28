import json
import mmcv

import os.path

with open('./train/bepro.json') as json_file:
    data = json.load(json_file)

annotations = []
images = []
obj_count = 0
idx = 0

for match_name,img_list in data["train"].items():
    print (match_name)

    with open(img_list) as f:
        for path in f:
            filename = '_'.join(path.strip().split('/')[1:])
            img_path = os.path.join(data["root"], path.strip())
            height, width = mmcv.imread(img_path).shape[:2]

            images.append(dict(
                id=idx,
                file_name=filename,
                height=height,
                width=width))
            
            idx+=1

            # reading label 
            lbl_path_lst = img_path.split('/')
            lbl_filename = lbl_path_lst[-1].split('.')[0]
            lbl_path_lst[3] = 'labels_with_ids'
            lbl_path_lst[-1] = '%s.txt' % lbl_filename
            lbl_path = '/'.join(lbl_path_lst)
            print (lbl_path)

            with open(lbl_path) as fl:
                for lb in fl:
                    print (lb.strip())

                    
            # bboxes = []
            # labels = []
            # masks = []
            # for _, obj in v['regions'].items():
            #     assert not obj['region_attributes']
            #     obj = obj['shape_attributes']
            #     px = obj['all_points_x']
            #     py = obj['all_points_y']
            #     poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            #     poly = [p for x in poly for p in x]

            #     x_min, y_min, x_max, y_max = (
            #         min(px), min(py), max(px), max(py))


            #     data_anno = dict(
            #         image_id=idx,
            #         id=obj_count,
            #         category_id=0,
            #         bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
            #         area=(x_max - x_min) * (y_max - y_min),
            #         segmentation=[poly],
            #         iscrowd=0)
            #     annotations.append(data_anno)
            #     obj_count += 1