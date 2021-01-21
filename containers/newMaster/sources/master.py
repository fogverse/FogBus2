import logging
import sys

from logger import get_logger
from registry import Registry
from connection import Server, Message
from node import Node
from datatype import User


class Master(Node):

    def __init__(
            self,
            myAddr,
            masterAddr,
            loggerAddr,
            masterID: int = 0,
            logLevel=logging.DEBUG):

        super().__init__(
            myAddr=myAddr,
            masterAddr=masterAddr,
            loggerAddr=loggerAddr,
            logLevel=logLevel
        )

        self.id = masterID
        self.registry: Registry = Registry(logLevel=logLevel)

    def run(self):
        self.role = 'master'
        self.name = 'Master-%d' % self.id
        self.logger = get_logger(self.name, self.logLevel)
        self.logger.info("Serving ...")

    def handleMessage(self, message: Message):
        if message.type == 'register':
            self.__handleRegister(message=message)
        elif message.type == 'data':
            self.__handleData(message=message)
        elif message.type == 'result':
            self.__handleResult(message=message)
        elif message.type == 'lookup':
            self.__handleLookup(message=message)
        elif message.type == 'ready':
            self.__handleReady(message=message)

    def __handleRegister(self, message: Message):
        respond = self.registry.register(message=message)
        self.sendMessage(respond, message.source.addr)
        self.logger.info('%s registered', respond['name'])

        if message.content['role'] == 'user':
            while self.registry.messageForWorker.qsize():
                msg, addr = self.registry.messageForWorker.get()
                self.sendMessage(msg, addr)

    def __handleData(self, message: Message):
        userID = message.content['userID']
        user = self.registry.users[userID]
        if not user.addr == message.source.addr:
            return

        for taskName in user.entranceTasksByName:
            taskHandlerToken = user.taskNameTokenMap[taskName].token
            taskHandler = self.registry.taskHandlerByToken[taskHandlerToken]
            self.sendMessage(message.content, taskHandler.addr)

    def __handleResult(self, message: Message):
        userID = message.content['userID']
        user = self.registry.users[userID]
        self.sendMessage(message.content, user.addr)

    def __handleLookup(self, message: Message):
        taskHandlerToken = message.content['token']
        self.logger.info('taskHandlerByToken: ', self.registry.taskHandlerByToken)
        if taskHandlerToken not in self.registry.taskHandlerByToken:
            return
        taskHandler = self.registry.taskHandlerByToken[taskHandlerToken]
        respond = {
            'type': 'taskHandlerAddr',
            'addr': taskHandler.addr,
            'token': taskHandlerToken
        }
        self.sendMessage(respond, message.source.addr)

    def __handleReady(self, message: Message):
        if not message.source.role == 'taskHandler':
            return

        taskHandlerToken = message.content['token']
        taskHandler = self.registry.taskHandlerByToken[taskHandlerToken]
        user = taskHandler.user
        user.taskHandlerByTaskName[taskHandler.taskName] = taskHandler

        if not len(user.taskNameTokenMap) == len(user.taskHandlerByTaskName):
            return

        msg = {'type': 'ready'}
        self.sendMessage(msg, user.addr)


if __name__ == '__main__':
    myAddr_ = (sys.argv[1], int(sys.argv[2]))
    masterAddr_ = (sys.argv[3], int(sys.argv[4]))
    loggerAddr_ = (sys.argv[5], int(sys.argv[6]))
    master_ = Master(
        myAddr=myAddr_,
        masterAddr=masterAddr_,
        loggerAddr=loggerAddr_
    )
    master_.run()
