import base64
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils

if __name__ == '__main__':
    # 1. 这里虽然文件夹名字还叫 JPEGImages (为了兼容某些模型读取逻辑)，
    #    但我们实际存进去的将会是 PNG 图片。
    jpgs_path   = "datasets/JPEGImages"  
    pngs_path   = "datasets/SegmentationClass"
    
    # 2. 请确保这里包含了你所有的类别
    classes     = ["_background_", "coal", "rock"]
    
    # 确保输出文件夹存在
    if not os.path.exists(jpgs_path):
        os.makedirs(jpgs_path)
    if not os.path.exists(pngs_path):
        os.makedirs(pngs_path)

    count = os.listdir("./datasets/before/") 
    for i in range(0, len(count)):
        path = os.path.join("./datasets/before", count[i])

        # 只处理 json 文件
        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))
            
            # --- 图像读取逻辑 ---
            if data['imageData']:
                imageData = data['imageData']
            else:
                # 如果json里没有base64数据，尝试寻找对应的图片文件
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)
            
            # --- 标签处理逻辑 ---
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))
            
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
            
            # --- 修改点：保存原图为 PNG ---
            # 这里的 .split(".")[0] 是为了去掉 .json 后缀
            # 然后加上 .png 后缀保存
            PIL.Image.fromarray(img).save(osp.join(jpgs_path, count[i].split(".")[0] + '.png'))

            # --- 保存标签图 (Mask) ---
            new = np.zeros([np.shape(img)[0],np.shape(img)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                new = new + index_all*(np.array(lbl) == index_json)

            utils.lblsave(osp.join(pngs_path, count[i].split(".")[0] + '.png'), new)
            
            # 打印日志提示已保存为 PNG
            print('Saved ' + count[i].split(".")[0] + '.png and label png')