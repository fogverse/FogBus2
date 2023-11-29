import cv2
import numpy as np
import traceback

from io import BytesIO
from pathlib import Path
from time import time
from threading import Thread

from .base import ApplicationUserSide
from ...component.basic import BasicComponent

frame_idx = 1
cnt = 1

class SmartCCTV(ApplicationUserSide):

    def __init__(
            self,
            videoPath: str,
            targetHeight: int,
            showWindow: bool,
            basicComponent: BasicComponent):
        super().__init__(
            appName='SmartCCTV',
            videoPath=videoPath,
            targetHeight=targetHeight,
            showWindow=showWindow,
            basicComponent=basicComponent)
        self.videoPath = Path(self.videoPath)

    def prepare(self):
        self.canStart.wait()

    def __decompress_img(self, bbytes):
        f = BytesIO(bbytes)
        img_arr = np.load(f, allow_pickle=True)
        return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    def __compress_img(self, img):
        _, encoded = cv2.imencode(f'.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 95))
        f = BytesIO()
        np.save(f, encoded)
        return f.getvalue()

    def __get_frame(self):
        global frame_idx
        try:
            while True:
                ret, frame = self.sensor.read()
                if not ret:
                    break
                frame = self.__compress_img(frame)
                now = time()
                inputData = (frame, frame_idx, now, False)
                self.dataToSubmit.put(inputData)
                if frame_idx % 500 == 0:
                    print('Got frame:')
                    print(now, frame_idx)
                frame_idx += 1
            print('Outside loop')
            inputData = (None, frame_idx, None, True)
            self.dataToSubmit.put(inputData)
            self.basicComponent.debugLogger.info(
                "[*] Sent all the frames and waiting for result ...")
        except Exception as e:
            traceback.print_exc()
        finally:
            print('Done get frames')

    def _run(self):
        global cnt
        self.basicComponent.debugLogger.info(f"[*] Start at {time()}")
        self.basicComponent.debugLogger.info("[*] Sending frames ...")
        Thread(target=self.__get_frame).run()

        videoOutput = cv2.VideoWriter(
            f'result/{self.videoPath.stem}-result{self.videoPath.suffix}',
            cv2.VideoWriter_fourcc(*'mp4v'), 25, (1920,1080))

        isLastFrame = False
        print('Listen to results:')
        while not isLastFrame:
            resultFrame, frame_idx, frameTime, isLastFrame = \
                self.resultForActuator.get()
            if frame_idx % 500 == 0:
                print('Got result frame:')
                print(frame_idx, frameTime, isLastFrame)
            if isLastFrame:
                print('isLastFrame', isLastFrame, 'done, frame_idx:', frame_idx)
                print('actual frames:', cnt - 1)
                break
            resultFrame = self.__decompress_img(resultFrame)
            responseTime = (time() - frameTime) * 1000
            if frame_idx % 500 == 0:
                print('response time', responseTime)
            self.responseTime.update(responseTime)
            self.responseTimeCount += 1
            videoOutput.write(resultFrame)
            cnt += 1
        videoOutput.release()
        self.sensor.release()
        self.basicComponent.debugLogger.info(f"[*] End at {time()}")
