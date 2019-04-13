# -*- coding: utf-8 -*-
"""
@author: lywen
"""
import io
import json
import time
import base64
from flask import Flask, render_template, request
from PIL import Image
import model
from apphelper.image import union_rbox,adjust_box_to_origin
import numpy as np

app = Flask(__name__)
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/api/ocr/general_ocr", methods=['POST'])
def general_ocr():
    upload_image = request.form['image']
    # print(upload_image)
    # 是否进行方向检测
    detect_direction = request.form['detect_direction']
    # print(detect_direction)
    image = base64.b64decode(upload_image)


    image = np.fromstring(image, np.uint8)
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = io.BytesIO(image)
    image = Image.open(image).convert('RGB')
    image = np.array(image)
    # print(image)


    '''
    jobid = uuid.uuid1().__str__()
    path = 'test/{}.jpg'.format(jobid)
    with open(path, 'wb') as f:
        f.write(image)
    img = cv2.imread(path)  ##GBR
    '''

    timeTake = time.time()
    _, result, angle = model.model(image,
                                   detectAngle = detect_direction,  ##是否进行文字方向检测，通过web传参控制
                                   config=dict(MAX_HORIZONTAL_GAP=200,  ##字符之间的最大间隔，用于文本行的合并
                                               MIN_V_OVERLAPS=0.6,
                                               MIN_SIZE_SIM=0.6,
                                               TEXT_PROPOSALS_MIN_SCORE=0.2, # 是文字的概率
                                               TEXT_PROPOSALS_NMS_THRESH=0.3,
                                               TEXT_LINE_NMS_THRESH=0.99,  ##文本行之间测iou值
                                               ),
                                   leftAdjust=True,  ##对检测的文本行进行向左延伸
                                   rightAdjust=True,  ##对检测的文本行进行向右延伸
                                   alph=0.1,  ##对检测的文本行进行向右、左延伸的倍数
                                   )
    result = union_rbox(result, 0.2)
    res = [{'text': x['text'],
            'name': str(i),
            'box': {'cx': x['cx'],
                    'cy': x['cy'],
                    'w': x['w'],
                    'h': x['h'],
                    'angle': x['degree']

                    }
            } for i, x in enumerate(result)]
    res = adjust_box_to_origin(image, angle, res)  ##修正box
    timeTake = time.time() - timeTake

    print({'res': res, 'timeTake': round(timeTake, 4)})
    return json.dumps({'res': res, 'timeTake': round(timeTake, 4)}, ensure_ascii=False)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8090)
