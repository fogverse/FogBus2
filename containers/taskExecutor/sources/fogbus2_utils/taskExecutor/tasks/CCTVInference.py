from .base import BaseTask

import cv2
import os
import torch

import numpy as np

from io import BytesIO

class CCTVInference(BaseTask):
    def __init__(self):
        super().__init__(taskID=112, taskName='CCTVInference')
        # The repository is taken from
        # https://github.com/WongKinYiu/yolov7 from commit
        # 84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca
        self.model = torch.hub.load('yolov7', 'custom',
                                    'yolo7tinycrowdhuman.pt', source='local')

    def __decompress_img(self, bbytes):
        f = BytesIO(bbytes)
        img_arr = np.load(f, allow_pickle=True)
        return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    def __compress_img(self, img):
        _, encoded = cv2.imencode(f'.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 95))
        f = BytesIO()
        np.save(f, encoded)
        return f.getvalue()

    def exec(self, data):
        frame, frame_idx, _time, isLastFrame = data
        if isLastFrame:
            return data
        frame = self.__decompress_img(frame)

        result = self.model(frame)
        if frame_idx % 500 == 0:
            result.print()
            print(frame_idx)
        result.render()

        frame = self.__compress_img(frame)
        return (frame, frame_idx, _time, isLastFrame)
