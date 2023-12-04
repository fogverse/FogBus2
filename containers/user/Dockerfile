ARG DREG
FROM ${DREG}ariqbasyar/fogbus2-fogverse:user-0

WORKDIR /workplace/fogbus2-app
COPY ./sources/requirementsDockerImage.txt .
RUN pip install -r requirementsDockerImage.txt

# Hostname
RUN echo "User" > /etc/hostname

COPY ./sources/ .
