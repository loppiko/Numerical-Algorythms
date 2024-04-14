from enum import Enum, auto


class Field(Enum):
    EMPTY = auto()
    EXIT = auto()
    OSK = auto()
    DUMPSTER = auto()


class Edge():
    def __init__(self, edgeNumber: int, edgeType: Field = Field.EMPTY) -> None:
        self.edgeNumber = edgeNumber
        self.edgeType = edgeType

    def __eq__(self, value: object) -> bool:
        return (value.edgeNumber == self.edgeNumber and value.edgeType == self.edgeType)
    
    def __hash__(self) -> int:
        return hash((self.edgeNumber, self.edgeType))

    def __str__(self) -> str:
        return f"Edge(num: {self.edgeNumber}, type: {self.edgeType})"


class Path():
    def __init__(self, startEdge: Edge, endEdge: Edge, weight: float) -> None:
        self.startEdge = startEdge
        self.endEdge = endEdge
        self.weight = weight

    def __str__(self) -> str:
        return f"Path(start: {self.startEdge.edgeNumber}, end: {self.endEdge.edgeNumber}, weight: {self.weight})"
    