#!/usr/bin/env python
# -*- coding:utf-8 -*-

__Author__ = 'Lin Xin'


class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.color='white'

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

    def change_color(self,color):
        self.color=color


class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self, n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self, n):
        return n in self.vertList

    def addEdge(self, f, t, cost=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], cost)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

    def knightTour(self, id, path, current_length, target_path_length):
        start_v = self.getVertex(id)
        start_v.change_color('grey')
        path.append(id)
        start_v.


def knightGraph(bdSize):
    ktGraph = Graph()
    for row in range(bdSize):
        for col in range(bdSize):
            nodeId = posToNodeId(row, col, bdSize)
            newPositions = genLegalMoves(row, col, bdSize)
            for e in newPositions:
                nid = posToNodeId(e[0], e[1], bdSize)
                ktGraph.addEdge(nodeId, nid)
    return ktGraph


def genLegalMoves(row, column, board_size):
    return [(row + i, column + j) for i in (-1,1,-2,2) for j in (-1,1,-2,2) if
            i!=j and i+j != 0 and 0 <= row + i < board_size and 0 <= column + j < board_size]


def posToNodeId(row, column, board_size):
    return (row * board_size) + column



