import json
import os
from typing import Dict
from copy import deepcopy
from edge import Edge
from resourcesInfo import ResourcesInfo, WorkerInfo


class PersistentStorage:

    def __init__(
            self,
            toFile: bool = True,
            folder: str = 'profiler'):
        self.__toFile: bool = toFile
        self.__folder: str = folder

        if not os.path.exists(folder):
            os.mkdir(folder)

    def write(self, filename: str, _dict: Dict):
        _dictCopy = deepcopy(_dict)

        for key, value in _dictCopy.items():
            _dictCopy[key] = self.__covert(value)

        if self.__toFile:
            self.__writeDictToFileInJson(filename, _dictCopy)

    def read(self, name: str):
        filename = name + '.json'

        content = self.__readFromFileInJson(filename)

        if content == {}:
            return content

        if name == 'edges':
            return self.__recoverObject(content, Edge)

        if name == 'nodeResources':
            return self.__recoverObject(content, ResourcesInfo)

        if name == 'imagesAndRunningContainers':
            res = self.__recoverObject(content, WorkerInfo)
            for k, v in res.items():
                for kk, vv in v.__dict__.items():
                    v.__dict__[kk] = set(vv)
                res[k] = v
            return res

        return content

    @staticmethod
    def __recoverObject(content, objType):
        res = {}

        for k, v in content.items():
            res[k] = objType()
            res[k].__dict__ = v
        return res

    @staticmethod
    def __covert(obj):
        if isinstance(obj, Edge):
            return dict(obj)
        if isinstance(obj, ResourcesInfo):
            return dict(obj)
        if isinstance(obj, WorkerInfo):
            objDict = dict(obj)
            for k, v in objDict.items():
                objDict[k] = list(v)
            return objDict
        return obj

    def __writeDictToFileInJson(self, filename: str, content: Dict):
        with open(
                os.path.join(self.__folder, filename),
                'w') as outfile:
            json.dump(content, outfile)

    def __readFromFileInJson(self, filename: str) -> Dict:
        if not os.path.exists(filename):
            return {}
        with open(
                os.path.join(self.__folder, filename),
                'r') as outfile:
            return json.loads(outfile.read())
