import logging
import sys
from logger import get_logger
from registry import Registry
from connection import Message, Identity
from typing import Tuple

Address = Tuple[str, int]


class Master(Registry):

    def __init__(
            self,
            myAddr,
            masterAddr,
            loggerAddr,
            schedulerName: str = None,
            masterID: int = 0,
            logLevel=logging.DEBUG):
        Registry.__init__(
            self,
            myAddr=myAddr,
            masterAddr=masterAddr,
            loggerAddr=loggerAddr,
            ignoreSocketErr=True,
            schedulerName=schedulerName,
            logLevel=logLevel)
        self.id = masterID

    def run(self):
        self.role = 'Master'
        self.setName()
        self.logger = get_logger(
            logger_name=self.nameLogPrinting,
            level_name=self.logLevel)
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
        elif message.type == 'exit':
            self.__handleExit(message=message)
        elif message.type == 'profiler':
            self.__handleProfiler(message=message)

    def __handleRegister(self, message: Message):
        respond = self.registerClient(message=message)
        if respond is None:
            return self.__stopClient(
                message.source,
                'No such role: %s' % message.content['role'])
        self.sendMessage(respond, message.source.addr)
        self.logger.info('%s registered', respond['nameLogPrinting'])

    def __handleData(self, message: Message):
        userID = message.content['userID']
        if userID not in self.users:
            return self.__stopClient(
                message.source,
                'User-%d does not exist' % userID)
        user = self.users[userID]
        if not user.addr == message.source.addr:
            return self.__stopClient(
                message.source,
                'You are not User-%d' % userID)

        for taskName in user.entranceTasksByName:
            taskHandlerToken = user.taskNameTokenMap[taskName].token
            taskHandler = self.taskHandlerByToken[taskHandlerToken]
            self.sendMessage(message.content, taskHandler.addr)

    def __handleResult(self, message: Message):
        userID = message.content['userID']
        if userID not in self.users:
            return self.__stopClient(
                message.source,
                'User-%d does not exist' % userID)
        user = self.users[userID]
        self.sendMessage(message.content, user.addr)

    def __handleLookup(self, message: Message):
        taskHandlerToken = message.content['token']
        if taskHandlerToken not in self.taskHandlerByToken:
            return
        taskHandler = self.taskHandlerByToken[taskHandlerToken]
        respond = {
            'type': 'taskHandlerInfo',
            'addr': taskHandler.addr,
            'token': taskHandlerToken
        }
        self.sendMessage(respond, message.source.addr)

    def __handleReady(self, message: Message):
        if not message.source.role == 'TaskHandler':
            return self.__stopClient(
                message.source,
                'You are not TaskHandler')

        taskHandlerToken = message.content['token']
        taskHandler = self.taskHandlerByToken[taskHandlerToken]
        taskHandler.ready.set()

        user = taskHandler.user
        user.lock.acquire()
        user.taskHandlerByTaskName[taskHandler.taskName] = taskHandler
        if len(user.taskNameTokenMap) == len(user.taskHandlerByTaskName):
            for taskName, taskHandler in user.taskHandlerByTaskName.items():
                if not taskHandler.ready.is_set():
                    user.lock.release()
                    return
            if not user.isReady:
                msg = {'type': 'ready'}
                self.sendMessage(msg, user.addr)
                user.isReady = True
        user.lock.release()

    def __handleExit(self, message: Message):
        self.logger.info(
            '%s at %s exit with reason: %s',
            message.source.nameLogPrinting,
            str(message.source.addr),
            message.content['reason'])

        response = {
            'type': 'stop',
            'reason': 'Your asked for. Reason: %s' % message.content['reason']}
        if message.source.role == 'user':
            if message.source.id not in self.users:
                return
            user = self.users[message.source.id]
            self.sendMessage(response, user.addr)
            msg = {'type': 'stop', 'reason': 'Your User has exited.'}
            for taskHandler in user.taskHandlerByTaskName.values():
                self.sendMessage(msg, taskHandler.addr)
            del self.users[message.source.id]
        elif message.source.role == 'TaskHandler':
            if message.source.id not in self.taskHandlers:
                return
            taskHandler = self.taskHandlers[message.source.id]
            self.sendMessage(response, taskHandler.addr)
            del self.taskHandlerByToken[taskHandler.token]
            del self.taskHandlers[message.source.id]
        elif message.source.role == 'worker':
            if message.source.id not in self.workers:
                return
            del self.workers[message.source.id]
            del self.workers[message.source.machineID]
        self.sendMessage(response, message.source.addr)

    def __handleProfiler(self, message: Message):
        profilers = message.content['profiler']
        # Merge
        self.edges = {**self.edges, **profilers[0]}
        self.nodeResources = {**self.nodeResources, **profilers[1]}
        self.averageProcessTime = {**self.averageProcessTime, **profilers[2]}
        self.averageRespondTime = {**self.averageRespondTime, **profilers[3]}
        self.imagesAndRunningContainers = {**self.imagesAndRunningContainers, **profilers[4]}

        # update
        self.scheduler.edges = self.edges
        self.scheduler.averageProcessTime = self.averageProcessTime

    def __stopClient(self, identity: Identity, reason: str = 'No reason'):
        msg = {'type': 'stop', 'reason': reason}
        self.sendMessage(msg, identity.addr)


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
