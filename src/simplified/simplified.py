import numpy as np


class SimplfiedSimulation:
    def __init__(self, n: int, s: int):
        self.__n = n
        self.__s = s

    
    def __performOneStep(self) -> bool:
        return np.random.choice([False, True])
    

    def start(self, numberOfTries: int) -> float:
        countOfExit = 0

        for _ in range(numberOfTries):
            n, s = self.__n, self.__s

            while n > 0 and s > 0:
                if (self.__performOneStep()):
                    n += 1
                    s -= 1
                else:
                    n -= 1
                    s += 1
            
            if (s == 0):
                countOfExit += 1

        return countOfExit / numberOfTries

            
if (__name__ == "__main__"):
    simplified = SimplfiedSimulation(2, 3)
    print(f"Percent of safe exit: {simplified.start(100_000)}")
    