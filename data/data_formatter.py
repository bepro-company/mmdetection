import json
import mmcv

import os.path
import cv2

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
            image = mmcv.imread(img_path)
            height, width = image.shape[:2]

            images.append(dict(
                id=idx,
                file_name=filename,
                height=height,
                width=width))
            
            # reading label 
            lbl_path_lst = img_path.split('/')
            lbl_filename = lbl_path_lst[-1].split('.')[0]
            lbl_path_lst[3] = 'labels_with_ids'
            lbl_path_lst[-1] = '%s.txt' % lbl_filename
            lbl_path = '/'.join(lbl_path_lst)
            print (lbl_path)

            bboxes = []
            labels = []
            masks = []

            with open(lbl_path) as fl:
                for lb in fl:
                    lb_lst = lb.strip().split(' ')

                    x, y, bw, bh = float(lb_lst[2]), float(lb_lst[3]), float(lb_lst[4]), float(lb_lst[5])

                    x *= width 
                    y *= height 
                    bw *= width 
                    bh *= height

                    if 0:
                        dump_path = "./d_%s.jpg" % (lbl_filename)
                        cv2.rectangle(image, (int(x), int(y)), (int(x+bw), int(y+bh)), (255,0,0), 2)
                        cv2.imwrite(dump_path, image)

                    data_anno = dict(
                        image_id=idx,
                        id=obj_count,
                        category_id=0,
                        bbox=[x, y, bw, bh],
                        area=bw * bh,
                        segmentation=[],
                        iscrowd=0)

                    annotations.append(data_anno)
                    obj_count += 1

            idx+=1

            coco_format_json = dict(
                images=images,
                annotations=annotations,
                categories=[{'id':0, 'name': 'player'}])
        
            mmcv.dump(coco_format_json, out_file)


