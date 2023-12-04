ARG DREG
FROM ${DREG}ariqbasyar/fogbus2-fogverse:CCTVInference-0

WORKDIR /workplace/fogbus2-app
COPY ./sources/requirements.txt .
RUN pip install -r requirements.txt
RUN git clone https://github.com/WongKinYiu/yolov7.git
RUN cd yolov7 && git checkout 84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca

# Hostname
RUN echo "CCTVInference" > /etc/hostname

COPY ./sources/ .
