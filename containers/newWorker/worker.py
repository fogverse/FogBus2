import socketio
import cv2
import threading
from queue import Queue
from datatype import NodeSpecs
from message import Message

sio = socketio.Client()


@sio.event(namespace='/registry')
def connect():
    print('connection established')


@sio.event(namespace='/registry')
def task(data):
    print("task comes")


@sio.event(namespace='/registry')
def disconnect():
    print('disconnected from server')


sio.connect('http://127.0.0.1:5000', namespaces=['/registry', 'task'])
print(sio.connection_namespaces)
msg = {"role": "worker", "nodeSpecs": NodeSpecs(1, 1, 1, 1)}
sio.emit('register', Message.encrypt(msg), namespace='/registry')

sio.wait()
