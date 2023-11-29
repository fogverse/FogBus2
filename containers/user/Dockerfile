FROM ariqbasyar/fogverse:input

WORKDIR /workplace/fogbus2-app
COPY ./sources/requirementsDockerImage.txt .
RUN pip install -r requirementsDockerImage.txt

# Hostname
RUN echo "User" > /etc/hostname

COPY ./sources/ .

# Run User
ENTRYPOINT ["python", "-u", "user.py"]
