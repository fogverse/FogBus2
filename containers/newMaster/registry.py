import logging

from threading import Lock
from queue import Queue
from logger import get_logger
from datatype import Worker, User, NodeSpecs
from typing import NoReturn
from datatype import Client
from collections import defaultdict


class Registry:

    def __init__(self, logLevel=logging.DEBUG):
        self.__currentWorkerID: int = 0
        self.__lockCurrentWorkerID: Lock = Lock()
        self.workers: dict[int, Worker] = {}
        self.__currentUserID: int = 0
        self.__lockCurrentUserID: Lock = Lock()
        self.users: dict[int, User] = {}
        self.clientBySocketID: dict[int, Client] = {}
        self.waitingWorkers: Queue[Worker] = Queue()

        self.workersQueueByAppID: dict[int, Queue[Worker]] = defaultdict(Queue[Worker])
        self.logger = get_logger('Master-Registry', logLevel)

    def register(self, client: Client, message: dict) -> Client:
        role = message['role']
        if role == 'user':
            return self.__addUser(client)
        elif role == 'worker':
            return self.__addWorker(client, message)

    def __addWorker(self, client: Client, message: dict) -> Worker:
        nodeSpecs: NodeSpecs = message['nodeSpecs']
        self.__lockCurrentWorkerID.acquire()
        self.__currentWorkerID += 1
        workerID = self.__currentWorkerID
        self.__lockCurrentWorkerID.release()
        worker = Worker(
            socketID=client.socketID,
            socket_=client.socket,
            sendingQueue=client.sendingQueue,
            receivingQueue=client.receivingQueue,
            workerID=workerID,
            specs=nodeSpecs
        )
        self.workers[workerID] = worker
        self.clientBySocketID[client.socketID] = worker
        self.workerWait(worker, appIDs=message['appIDs'])
        self.logger.info("Worker-%d added. %s", workerID, worker.specs.info())
        return worker

    def workerWait(self, worker: Worker, appID: int = None, appIDs: dict = None) -> NoReturn:
        if appID is not None:
            self.workersQueueByAppID[appID].put(worker)
        else:
            for appID in appIDs:
                self.workersQueueByAppID[appID].put(worker)

    def workerWork(self, appID: int) -> Worker:
        while True:
            worker = self.workersQueueByAppID[appID].get(block=False)
            if not worker.active:
                continue
            else:
                return worker

    def removeWorker(self, workerID) -> NoReturn:
        self.workers[workerID].active = False

    def __addUser(self, client: Client) -> User:
        self.__lockCurrentUserID.acquire()
        self.__currentUserID += 1
        userID = self.__currentUserID
        self.__lockCurrentUserID.release()
        user = User(socketID=client.socketID,
                    socket_=client.socket,
                    receivingQueue=client.receivingQueue,
                    sendingQueue=client.sendingQueue,
                    userID=userID)
        self.users[userID] = user
        self.clientBySocketID[client.socketID] = user
        self.logger.info("User-%d added.", userID)
        return user

    def removeUser(self, userID):
        del self.users[userID]
