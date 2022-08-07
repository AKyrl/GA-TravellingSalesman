import csv

import Reporter
import numpy as np
import sys
from random import sample


# Modify the class name to match your student number.
class r0785071:
    populationSize = 200

    k_selection = 8
    k_elimination = 8

    mutationProbability_init = 0.1
    percentageOfSwitches_init = 0.1
    crossoverProbability_init = 1

    iterations = 10
    genForConvergence = 5
    stoppingConvergenceSlope = 0.000001

    printEveryIter = True

    class Individual:

        def __init__(self, numberOfCities, distanceMatrix, mutationProbability_init, numberOfSwitches_init, crossoverProbability_init):
            self.numberOfSwitches = numberOfSwitches_init
            self.mutationProbability = mutationProbability_init
            self.crossoverProbability = crossoverProbability_init
            self.cost = None
            self.path = np.random.permutation(np.arange(1, numberOfCities))
            self.setCost(distanceMatrix)
            self.numberOfCities = numberOfCities

        def setCost(self, distanceMatrix):
            total = distanceMatrix[0][self.path[0]]
            for i in range(len(self.path) - 1):
                c = distanceMatrix[self.path[i]][self.path[i + 1]]
                if c == np.inf:
                    total += sys.maxsize
                else:
                    total += c
            total += distanceMatrix[self.path[len(self.path) - 1]][0]
            self.cost = total

        def mutate(self, distanceMatrix):
            if np.random.rand() <= self.mutationProbability:
                self.mutate_path_randomSwaps()
                # also mutate other values for self Adaptability
                self.setCost(distanceMatrix)
            return

        def mutate_path_randomSwaps(self):
            for i in range(self.numberOfSwitches):
                index1 = np.random.randint(self.numberOfCities - 1)
                index2 = np.random.randint(self.numberOfCities - 1)
                temp = self.path[index1]
                self.path[index1] = self.path[index2]
                self.path[index2] = temp

    def __init__(self):

        self.population = None
        self.distanceMatrix = None
        self.numberOfCities = None
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def initialisation(self):
        population = np.ndarray(dtype=r0785071.Individual, shape=self.populationSize)
        numberOfSwitches_init = int(self.percentageOfSwitches_init * self.numberOfCities)
        for i in range(self.populationSize):
            population[i] = r0785071.Individual(self.numberOfCities, self.distanceMatrix, self.mutationProbability_init, numberOfSwitches_init, self.crossoverProbability_init)
        self.population = population

    def selection(self):
        return self.k_tournament(self.k_selection)

    def crossover(self, p1, p2):
        if np.random.rand() <= (p1.crossoverProbability + p2.crossoverProbability) / 2:
            return self.pmx_pair(p1, p2)
        return None

    def elimination(self):
        newPopulation = np.ndarray(self.populationSize, dtype=r0785071.Individual)
        for i in range(self.populationSize):
            newPopulation[i] = self.k_tournament(self.k_elimination)
        self.population = newPopulation

    def stoppingCriteria(self, means, index):
        flag = True
        if index > self.genForConvergence:
            indexes = np.arange(float(self.genForConvergence))
            slope = np.polyfit(indexes, means, 1)
            if self.printEveryIter:
                print("slope: ", slope[0] / np.mean(means))
            if abs(slope[0] / np.mean(means)) < self.stoppingConvergenceSlope:
                print("slope: ", slope[0] / np.mean(means), "lastMeans: ", means)
                flag = False
        return flag

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        self.distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        self.numberOfCities = len(self.distanceMatrix)

        # Initialisation
        self.initialisation()

        (meanObjective, bestObjective, bestSolution) = self.accessQualityOfGeneration()
        lastMeans = np.zeros(self.genForConvergence)

        f = open('plot.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'MeanValue', 'BestValue'])

        iteration = 0
        lastMeans = self.addNewMean(lastMeans, meanObjective)

        while iteration < self.iterations and self.stoppingCriteria(lastMeans, iteration):

            # mutation
            for ind in self.population:
                ind.mutate(self.distanceMatrix)

            # crossover
            offsprings = np.ndarray(self.populationSize, dtype=r0785071.Individual)
            nbr_offspring = 0
            for j in range(self.populationSize // 2):
                p1 = self.selection()
                p2 = self.selection()
                new_individuals = self.crossover(p1, p2)
                if new_individuals is not None:
                    offsprings[nbr_offspring] = new_individuals[0].mutate
                    offsprings[nbr_offspring + 1] = new_individuals[1].mutate
                    nbr_offspring += 2
            offsprings.resize(nbr_offspring)

            self.population = np.concatenate((self.population, offsprings))

            # elimination
            self.elimination()

            (meanObjective, bestObjective, bestSolution) = self.accessQualityOfGeneration()
            if self.printEveryIter:
                print("meanObjective:", meanObjective, ", bestObjective:", bestObjective, "diff:",
                      meanObjective - bestObjective)
                # print(bestSolution)
                print("I:", iteration)

            # write a row to the csv file
            writer.writerow([iteration, meanObjective, bestObjective])

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

            iteration += 1
            lastMeans = self.addNewMean(lastMeans, meanObjective)

        # Your code here.
        return 0

    def accessQualityOfGeneration(self):
        # Values for each iteration
        fitness = np.ndarray(self.populationSize)
        for i in range(self.populationSize):
            fitness[i] = self.population[i].cost

        meanObjective = np.mean(fitness)
        bestObjective = np.min(fitness)
        bestSolution = self.population[np.argmin(fitness)].path.tolist()
        bestSolution.append(0)
        return meanObjective, bestObjective, np.array(bestSolution)

    def addNewMean(self, means, newMean):
        means = np.roll(means, 1)
        means[0] = newMean
        return means

    def k_tournament(self, k) -> Individual:
        random_index_Sample = sample(range(self.populationSize), k)
        costSample = np.ndarray(k)
        for i in range(k):
            ind = self.population[random_index_Sample[i]]
            costSample[i] = ind.cost

        best_index = np.argmin(costSample)

        return self.population[random_index_Sample[best_index]]

    def pmx_pair(self, p1 : Individual, p2 : Individual):
        a = p1.path
        b = p2.path

        half = len(a) // 2
        start = np.random.randint(0, len(a) - half)
        stop = start + half

        child1path = self.pmx(b, a, start, stop)
        child2path = self.pmx(a, b, start, stop)
        childMutationProbability = (p1.mutationProbability + p2.mutationProbability) / 2
        childCrossoveerProbability = (p1.crossoverProbability + p2.crossoverProbability) / 2
        childNumberOfSwitches = (p1.numberOfSwitches + p2.numberOfSwitches) / 2

        child1 = r0785071.Individual(p1.numberOfCities, self.distanceMatrix, childMutationProbability, childNumberOfSwitches, childCrossoveerProbability)
        child2 = r0785071.Individual(p1.numberOfCities, self.distanceMatrix, childMutationProbability,
                                     childNumberOfSwitches, childCrossoveerProbability)
        child1.path = child1path
        child2.path = child2path

        return child1, child2

    def pmx(self, a, b, start, stop): # https://www.uio.no/studier/emner/matnat/ifi/INF3490/h16/exercises/inf3490-sol2.pdf
        child = [None] * len(a)
        # Copy a slice from first parent:
        child[start:stop] = a[start:stop]
        # Map the same slice in parent b to child using indices from parent a:
        for ind, x in enumerate(b[start:stop]):
            ind += start
            if x not in child:
                while child[ind] != None:
                    # ind = b.index(a[ind])
                    ind = np.where(b == a[ind])[0][0]
                child[ind] = x
        # Copy over the rest from parent b
        for ind, x in enumerate(child):
            if x == None:
                child[ind] = b[ind]
        return np.array(child)


if __name__ == "__main__":
    r = r0785071()
    r.optimize("tour100.csv")
