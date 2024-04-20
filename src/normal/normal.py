import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from nodeTypes import Path, Edge, Field

from GaussSeidel import gauss_seidel
from GaussWithBase import gauss_elimination_with_partial_pivot
from GaussWithoutBase import gauss_Elimination

class NormalSimulation:

    allPaths: list[Path] = []
    allEdges: dict[Edge, list] = {}

    def readInput(self, fileName: str) -> None:
        data = []

        with open(fileName) as input:
            content = input.readlines()

        for line in content:
            data.append(line.rstrip("\n").split(" "))
        indexOfFieldsDescr = data.index([""])

        self.oskEdgeNumbers = [int(x) for x in data[indexOfFieldsDescr + 1][1:]]
        self.exitEdgeNumbers = [int(x) for x in data[indexOfFieldsDescr + 2][1:]]
        self.travelersStreetNumbers = [int(x) for x in data[indexOfFieldsDescr + 3][1:]]
        self.dumpEdgeNumbers = [int(x) for x in data[indexOfFieldsDescr + 4][1:]]


        for startEdgeNumber, endEdgeNumber, pathWeight in data[1:indexOfFieldsDescr]:
            startEdge = Edge(int(startEdgeNumber), self.defineEdgeType(int(startEdgeNumber)))
            endEdge = Edge(int(endEdgeNumber), self.defineEdgeType(int(endEdgeNumber)))

            self.allPaths.append(Path(startEdge, endEdge, int(pathWeight)))
            
            if startEdge not in self.allEdges.keys():
                self.allEdges[startEdge] = []
            if startEdge.edgeType == Field.EMPTY or startEdge.edgeType == Field.DUMPSTER:
                self.allEdges[startEdge].append(int(endEdgeNumber))
            
            if endEdge not in self.allEdges.keys():
                self.allEdges[endEdge] = []
            if endEdge.edgeType == Field.EMPTY or endEdge.edgeType == Field.DUMPSTER:
                self.allEdges[endEdge].append(int(startEdgeNumber))

        # print({str(key): value for key, value in self.allEdges.items()})

        self.generateMatrix()



    def defineEdgeType(self, edgeNum: int) -> Field:
        if edgeNum in self.oskEdgeNumbers:
            return Field.OSK
        elif edgeNum in self.exitEdgeNumbers:
            return Field.EXIT
        elif edgeNum in self.dumpEdgeNumbers:
            return Field.DUMPSTER
        else:
            return Field.EMPTY
        
        

    def generateMatrix(self) -> None:
        self.totalNumberOfFields = len(self.allEdges) + sum([x.weight for x in self.allPaths])
        self.resultMatrix = np.transpose(np.zeros(self.totalNumberOfFields))
        self.propabilityMatrix = []
        self.positionOfEdges = {}

        
        for path in self.allPaths:
            self.addEdgeToMatrix(path.startEdge)

            dumpsterModificator = 0 if (not path.startEdge.edgeType == Field.DUMPSTER) else 0.25
            if not path.startEdge in self.positionOfEdges:
                self.propabilityMatrix.append(self.calculateStandardRow(np.zeros(self.totalNumberOfFields), dumpsterModificator))
            else:
                self.propabilityMatrix.append(self.calculateStandardRow(np.zeros(self.totalNumberOfFields), dumpsterModificator, previousEdge=self.positionOfEdges[path.startEdge]))
                self.updateEdgeProbability(self.positionOfEdges[path.startEdge])


            for _ in range(path.weight - 2):
                self.propabilityMatrix.append(self.calculateStandardRow(np.zeros(self.totalNumberOfFields)))


            dumpsterModificator = 0 if (not path.endEdge.edgeType == Field.DUMPSTER) else 0.25
        
            if not path.endEdge in self.positionOfEdges and path.weight > 1:
                self.propabilityMatrix.append(self.calculateStandardRow(np.zeros(self.totalNumberOfFields), dumpsterModificator))
            elif path.weight > 1:
                self.propabilityMatrix.append(self.calculateStandardRow(np.zeros(self.totalNumberOfFields), dumpsterModificator, nextEdge=self.positionOfEdges[path.endEdge]))
                self.updateEdgeProbability(self.positionOfEdges[path.endEdge])

            self.addEdgeToMatrix(path.endEdge)

        np.savetxt("probabilityMatrix.txt", self.propabilityMatrix, fmt='%.2f')
        np.savetxt("probabilityResultMatrix.txt", self.resultMatrix, fmt='%.2f')

        gauss_seidel(self.propabilityMatrix, self.resultMatrix)
        gauss_elimination_with_partial_pivot(self.propabilityMatrix, self.resultMatrix)
        gauss_Elimination(self.propabilityMatrix, self.resultMatrix)

    
    def addEdgeToMatrix(self, currEdge: Edge) -> None:
        if (not currEdge in self.positionOfEdges.keys()):
            self.positionOfEdges[currEdge] = len(self.propabilityMatrix)
            currRow = np.zeros(self.totalNumberOfFields)
                
            if (not self.updateResultMatrix(currEdge)):
                if len(self.propabilityMatrix) > 0:
                    currRow[len(self.propabilityMatrix) - 1] = -0.5
                currRow[len(self.propabilityMatrix)] = 1
                if len(self.propabilityMatrix) + 1 < len(currRow):
                    currRow[len(self.propabilityMatrix) + 1] = -0.5
                self.propabilityMatrix.append(currRow)
            else:
                currRow[len(self.propabilityMatrix)] = 1
                self.propabilityMatrix.append(currRow)


    def updateEdgeProbability(self, edgeIndex: int) -> int:
        if (not np.all(list(map(lambda x: True if (x == 0.0 or x == 1.0) else False, self.propabilityMatrix[edgeIndex])))):
            currEdge = np.array(self.propabilityMatrix[edgeIndex])
            filteredIndicies = np.where((currEdge != 0.0) & (currEdge != 1.0))[0]

            for index in filteredIndicies:
                self.propabilityMatrix[edgeIndex][index] = -1.0 / (len(filteredIndicies) + 1)
            self.propabilityMatrix[edgeIndex][len(self.propabilityMatrix) - 1] = -1.0 / (len(filteredIndicies) + 1)



    def calculateStandardRow(self, zeros: list, dumpsterModifier: float = 0.0, previousEdge: int = None, nextEdge: int = None) -> list[float]:
        zeros[previousEdge if previousEdge is not None else len(self.propabilityMatrix) - 1] = -0.5 + dumpsterModifier
        zeros[len(self.propabilityMatrix)] = 1
        zeros[nextEdge if nextEdge is not None else len(self.propabilityMatrix) + 1] = -0.5 - dumpsterModifier
        return zeros


    def updateResultMatrix(self, currEdge: Edge) -> bool:
        match (currEdge.edgeType):
            case Field.OSK:
                return True
            case Field.EXIT:
                self.resultMatrix[len(self.propabilityMatrix)] = 1
                return True
            case _:
                return False
            
    
    def monteCarlo(self, edgeNumber: int, iterations: int) -> float:
        
        numberOfWins = 0
        
        for _ in range(iterations):
            
            startEdge, endEdge, currPath, currPosition = self.changePath(self.findEdge(edgeNumber), None, 0)

            while True:
                if (currPosition == 0):
                    if (startEdge.edgeType == Field.EXIT):
                        numberOfWins += 1
                    if (startEdge.edgeType == Field.OSK or startEdge.edgeType == Field.EXIT):
                        break
                    startEdge, endEdge, currPath, currPosition = self.changePath(startEdge, endEdge, currPosition)
                if (currPosition == currPath.weight + 1):
                    if (endEdge.edgeType == Field.EXIT):
                        numberOfWins += 1
                    if (endEdge.edgeType == Field.OSK or endEdge.edgeType == Field.EXIT):
                        break
                    startEdge, endEdge, currPath, currPosition = self.changePath(startEdge, endEdge, currPosition)
                else:
                    backwardsDumpsterModificator = 25 if (currPosition == 1 and startEdge.edgeType == Field.DUMPSTER) else 0
                    forwardDumpsterModificator = 25 if (currPosition == currPath.weight and endEdge.edgeType == Field.DUMPSTER) else 0
                    currPosition += 1 if (np.random.randint(100) < 50 + backwardsDumpsterModificator - forwardDumpsterModificator) else -1

        return numberOfWins / iterations
                        


    def changePath(self, startEdge: Edge, endEdge: Edge, currPosition: int) -> list[Edge, Edge, Path, int]:
        if currPosition == 0 or endEdge is None:
            endEdge = self.findEdge(np.random.choice(self.allEdges[startEdge]))
        else:
            startEdge = endEdge
            endEdge = self.findEdge(np.random.choice(self.allEdges[startEdge]))
        return startEdge, endEdge, Path(startEdge, endEdge, self.findPathWeight(startEdge.edgeNumber, endEdge.edgeNumber)), 1            


    def findEdge(self, edgeNumber: int) -> Edge:
        result = None

        for edge in list(self.allEdges.keys()):
            if (edge.edgeNumber == edgeNumber):
                result = edge
                break

        return result
    

    def findPathWeight(self, startEdgeNumber: int, endEdgeNumber: int) -> Path:
        for path in self.allPaths:
            if (path.startEdge.edgeNumber == startEdgeNumber and path.endEdge.edgeNumber == endEdgeNumber) or (path.endEdge.edgeNumber == startEdgeNumber and path.startEdge.edgeNumber == endEdgeNumber):
                result = path
                break
            
        return result.weight
    

    def randomiseInput(self, numberOfEdges: int, numberOfPaths: int, numberOfOSK: int, numberOfExit: int, numberOfDumpsters: int) -> None:
        randomisedInput = []
        edgesToAdd = [x + 1 for x in range(numberOfEdges)]

        randomisedInput.append([numberOfEdges, numberOfPaths])
        for _ in range(numberOfPaths):
            if (len(edgesToAdd) == 0):
                randomisedInput.append([*np.random.randint(1, numberOfEdges + 1, size=2), np.random.randint(1, 15)])
            else:
                firstEdge = edgesToAdd[0]
                secondEdge = np.random.randint(1, numberOfEdges + 1)
                while firstEdge == secondEdge:
                    secondEdge = np.random.randint(1, numberOfEdges + 1)
                edgesToAdd = edgesToAdd[1:]
                randomisedInput.append([firstEdge, secondEdge, np.random.randint(1, 15)])

        randomisedInput.append([])
        
        allTypedEdges = set()
        while len(allTypedEdges) < numberOfOSK+numberOfExit+numberOfDumpsters+1:
            randNumber = np.random.randint(1, numberOfEdges + 1)
            if (randNumber not in allTypedEdges):
                allTypedEdges.add(randNumber)

        allTypedEdges = list(allTypedEdges)

        randomisedInput.append([numberOfOSK, *allTypedEdges[0:numberOfOSK]])
        randomisedInput.append([numberOfExit, *allTypedEdges[numberOfOSK:numberOfOSK+numberOfExit]])
        randomisedInput.append([1, *allTypedEdges[numberOfOSK+numberOfExit:numberOfOSK+numberOfExit+1]])
        randomisedInput.append([numberOfDumpsters, *allTypedEdges[numberOfOSK+numberOfExit+1:numberOfOSK+numberOfExit+1+numberOfDumpsters]])
        print(randomisedInput)

        with open("inputRandom.conf", 'w') as file:
            for row in randomisedInput:
                file.write(" ".join(map(str, row)) + "\n")


    def drawGraph(self) -> None:
        G = nx.Graph()

        for path in self.allPaths:
            G.add_edge(path.startEdge.edgeNumber, path.endEdge.edgeNumber, weight=path.weight)

        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 5]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 5]

        pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility
        nx.draw_networkx_nodes(G, pos, node_size=700)   # nodes

        # edges
        nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
        nx.draw_networkx_edges(G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed")
        
        nx.draw_networkx_labels(G, pos, font_size=15, font_family="sans-serif") # node labels

        edge_labels = nx.get_edge_attributes(G, "weight")         # edge weight labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

        plt.axis("off")
        plt.savefig("graph.png")



if (__name__ == "__main__"):
    normal = NormalSimulation()
    # normal.randomiseInput(10, 15, 2, 2, 0)
    normal.readInput("input2.conf")
    # normal.drawGraph()
    # print(normal.monteCarlo(3, 100_000))