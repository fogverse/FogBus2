ARG DREG
FROM ${DREG}ariqbasyar/pytorch:v2.1.1-py3.9.18-yolov7

WORKDIR /workplace
COPY ./sources/requirements.txt .
RUN pip install -r requirements.txt
RUN git clone https://github.com/WongKinYiu/yolov7.git
RUN cd yolov7 && git checkout 84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca

# Hostname
RUN echo "CCTVInference" > /etc/hostname

COPY ./sources/ .

# Run NaiveFormula0
ENTRYPOINT ["python", "-u", "taskExecutor.py"]
