import json
import mmcv

import os.path
import cv2
import sys

from pathlib import Path

# with open('./train/bepro.json') as json_file:
with open('./%s/%s.json' % (sys.argv[1], sys.argv[2])) as json_file:
    data = json.load(json_file)

images = []
obj_count = 0
idx = 0
annotations = []

for match_name,img_list in data["train"].items():
    print ('processing match %s' % match_name)

    with open(img_list) as f:
        for path in f:
            print ('processing image %s' % path.strip())

            img_path = os.path.join('/mmdetection/data/%s/' % sys.argv[1], path.strip())
            # img_path = os.path.join(data["root"], path.strip())
            image = mmcv.imread(img_path)
            height, width = image.shape[:2]

            images.append(dict(
                id=idx,
                file_name=img_path,
                height=height,
                width=width))
            
            # reading label 
            lbl_path_lst = img_path.split('/')
            lbl_filename = lbl_path_lst[-1].split('.')[0]
            lbl_path_lst[4] = 'labels_with_ids'
            lbl_path_lst[-1] = '%s.txt' % lbl_filename
            lbl_path = '/'.join(lbl_path_lst)
            
            # print ('processing label file %s' % lbl_path)

            assert os.path.isfile(lbl_path) == True 

            with open(lbl_path) as fl:
                for lb in fl:
                    lb_lst = lb.strip().split(' ')

                    xc, yc, bw, bh = float(lb_lst[2]), float(lb_lst[3]), float(lb_lst[4]), float(lb_lst[5])

                    xc *= width 
                    yc *= height 
                    bw *= width 
                    bh *= height

                    x, y = xc-bw/2, yc-bh/2
                    # print (x,y,bw,bh)

                    if 0:
                        dump_path = "./%s_%s.jpg" % (match_name, lbl_filename)
                        cv2.rectangle(image, (int(x), int(y)), (int(x+bw), int(y+bh)), (255,0,0), 2)
                        cv2.imwrite(dump_path, image)

                    data_anno = dict(
                        image_id=idx,
                        id=obj_count,
                        category_id=0,
                        bbox=[x, y, bw, bh],
                        area=bw*bh,
                        segmentation=[],
                        iscrowd=0)

                    annotations.append(data_anno)
                    obj_count += 1

            # assert len(annotations) > 0
            idx+=1

coco_format_json = dict(
    images=images,
    annotations=annotations,
    categories=[{'id':0, 'name': 'player'}])

Path('./%s/labels_coco/' % sys.argv[1]).mkdir(parents=True, exist_ok=True)
mmcv.dump(coco_format_json, './%s/labels_coco/%s.json' % (sys.argv[1], 'bepro_coco'))


