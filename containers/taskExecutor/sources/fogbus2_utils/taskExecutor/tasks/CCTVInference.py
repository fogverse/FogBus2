from .base import BaseTask

import os
import torch

cnt = 1

class CCTVInference(BaseTask):
    def __init__(self):
        super().__init__(taskID=112, taskName='CCTVInference')
        # The repository is taken from
        # https://github.com/WongKinYiu/yolov7 from commit
        # 84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca
        self.model = torch.hub.load('yolov7', 'custom',
                                    'yolo7crowdhuman.pt', source='local')

    def exec(self, data):
        global cnt
        print('='*40, flush=True)
        print('New data', flush=True)
        frame, _time, isLastFrame = data
        if isLastFrame:
            return None
        result = self.model(frame)
        result.print()
        result.render()
        print(cnt)
        print('='*40, flush=True)
        cnt += 1
        return (frame, _time, isLastFrame)
