# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

################################
# Agent Technique Introduction #
################################

"""
    HEURISTIC SEARCH PACMAN AGENTS

    The technique we used in this implementation is Heuristic Search
    A* algorithm is used, with different goal and different heuristic under different circumstances

"""

import sys

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
import re, os


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'WaStarInvader', second = 'WaStarDefender'):

    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########


# Note: the following class is not used, but is kept for backwards
# compatibility with team submissions that try to import it.
class AgentFactory:
    "Generates agents for a side"

    def __init__(self, isRed, **args):
        self.isRed = isRed
        self.agent = True

    def getAgent(self, index):
        '''
        # "Returns the agent for the provided index."
        # util.raiseNotDefined()
        if self.isRed:
            # if index == "1":
            if index == 1:
                return WaStarInvader(index)
            # elif index == "3":
            elif index == 3:
                return WaStarDefender(index)
        else:
            # if index == "0":
            if index == 0:
                return WaStarInvader(index)
            # elif index == "2":
            elif index == 2:
                return WaStarDefender(index)
        '''
        if self.agent:
            self.agent = not self.agent
            return WaStarInvader(index)
        else:
            self.agent = not self.agent
            return WaStarDefender(index)

class DummyAgent(CaptureAgent):

    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    # Agent Mode
    # Under different mode, the Heuristic Search - A* Algorithm will use different goal and heuristic
    mode = ""

    # Safe Coordinates
    # Definition: Coordinates that have at least two paths which can reach home
    safeCoordinates = []

    # Risky Coordinates
    # Definition: Coordinates that are not classified as safe
    riskyCoordinates = []

    # Steps from the last food / capsule / ghost eaten (Used for testing only. )
    hungrySteps = 0

    # Before submitting the code, turn this to False to hide debug messages
    debug_message = True

    def __init__(self, index):
        super().__init__(index)

        # Layout related
        self.maze_dim = None
        self.boarder_mid = None
        self.boardWidth = None
        self.boardHeight = None
        self.pathDict = None

        # game state variables
        self.initialPosition = None
        self.currentPosition = None
        self.opponentIndices = None
        self.searching = None
        self.eatenFoods = None
        self.walls = None

    def isSafeCoordinate(self, coordinate, gameState):

        """
        Helper function to help decide the goal during heuristic search
        Decide whether the given coordinate is safe or not,
        i.e. whether there exists at least two ways back home.
        """

        wallList = gameState.getWalls().asList()
        if coordinate in wallList:
            return False

        capsuleList = self.getCapsuleList(gameState)
        for capsule in capsuleList:
            if capsule is not None and self.getMazeDistance(coordinate, capsule) <= 2:
                return True

        x, y = coordinate[0], coordinate[1]
        legalAction = self.getLegalActionOfPosition(coordinate, gameState)
        if len(legalAction) <= 1:
            return False
        if len(legalAction) == 2 and Directions.STOP in legalAction:
            return False

        nonStopLegalAction = []
        for action in legalAction:
            if action != Directions.STOP:
                nonStopLegalAction.append(action)

        newStartingPoint = []
        for action in nonStopLegalAction:
            if action == Directions.EAST:
                newStartingPoint.append((x + 1, y))
            elif action == Directions.WEST:
                newStartingPoint.append((x - 1, y))
            elif action == Directions.NORTH:
                newStartingPoint.append((x, y + 1))
            elif action == Directions.SOUTH:
                newStartingPoint.append((x, y - 1))

        numberOfEscapePath = 0
        for startingPoint in newStartingPoint:
            dfsProblem = PositionSearchProblem(gameState, startingPoint)
            boardWidth, boardHeight = self.getWidthandHeight(gameState)
            path = self.depthFirstSearch(dfsProblem, coordinate, self.red, boardWidth, self.getWallList(gameState), self.getOpponentList(gameState))
            if len(path) != 0:
                numberOfEscapePath += 1
            if numberOfEscapePath > 1:
                return True
        return False

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """
        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        '''
        Your initialization code goes here, if you need any.
        '''

        self.walls = gameState.getWalls()

        # dimensions of the grid world w * h
        self.maze_dim = (gameState.data.layout.width, gameState.data.layout.height)

        self.opponentIndices = self.getOpponents(gameState)

        self.boardWidth = self.maze_dim[0]
        self.boardHeight = self.maze_dim[1]

        """
        # Initialise safeCoordinates, used to decide goals during heuristic search
        """
        for x in range(self.boardWidth):
            for y in range(self.boardHeight):
                if self.isSafeCoordinate((x, y), gameState):
                    self.safeCoordinates.append((x, y))
                else:
                    self.riskyCoordinates.append((x, y))

        self.pathDict = self.scanMaze()

        if self.debug_message: print("==========Pre-computation Done==========")

    ####################
    # Helper Functions #
    ####################

    def scanMaze(self):
        """
        Scan through every reachable point in the given maze and calculate the shortest paths point-wise
        :return: a dictionary --> dict[source][target] = source-<...action...>-target
        """
        def getReversedDirection(actions):
            reversedActions = []
            for action in actions[::-1]:
                if action == Directions.NORTH: reversedActions.append(Directions.SOUTH)
                elif action == Directions.SOUTH: reversedActions.append(Directions.NORTH)
                elif action == Directions.WEST: reversedActions.append(Directions.EAST)
                elif action == Directions.EAST: reversedActions.append(Directions.WEST)
                else: reversedActions.append(Directions.STOP)
            return reversedActions

        def getDistanceOnMaze(walls):
            valid_points = [(x, y) for x in range(self.maze_dim[0]) for y in range(self.maze_dim[1]) if
                            (x, y) not in walls.asList()]
            path = {}
            for p1 in valid_points:
                open = util.Queue()
                if p1 not in path.keys():
                    path[p1] = {p1: {}}
                path[p1][p1] = []
                init = (p1, [])
                open.push(init)
                closed = []
                while len(closed) < len(valid_points):
                    currNode = open.pop()
                    currState = currNode[0]
                    currPath = currNode[1]
                    if currState not in closed:
                        successors = []
                        x, y = currState
                        if not walls[x][y + 1]:
                            successors.append( ((x, y + 1), Directions.NORTH) )
                        if not walls[x][y - 1]:
                            successors.append( ((x, y - 1), Directions.SOUTH) )
                        if not walls[x + 1][y]:
                            successors.append( ((x + 1, y), Directions.EAST) )
                        if not walls[x - 1][y]:
                            successors.append( ((x - 1, y), Directions.WEST) )
                        if len(successors) > 0:
                            for each in successors:
                                if currState not in path.keys(): path[currState] = {}
                                if each[0] not in path.keys(): path[each[0]] = {}
                                '''
                                BFS Speed-up. Infer backwards from forward path
                                Trick:
                                    1. Reduced search space
                                    2. Every one-step gives at most four
                                     > Two for adjacency
                                     > Two for init and one another
                                '''
                                if each[0] not in closed:
                                    path[currState][each[0]] = [each[1]]
                                    path[each[0]][currState] = getReversedDirection([each[1]])
                                    # assert len(path[currState][each[0]]) == len(path[each[0]][currState])
                                    temp = (each[0], currPath + [each[1]])
                                    path[p1][each[0]] = temp[1]
                                    path[each[0]][p1] = getReversedDirection(temp[1])
                                    # assert len(path[p1][each[0]]) == len(path[each[0]][p1])
                                    open.push(temp)
                        closed.append(currState)
            return path

        return getDistanceOnMaze(self.walls)

    def chooseAction(self, gameState):
        pass

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        position = successor.getAgentState(self.index).getPosition()
        if position != util.nearestPoint(position):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getFeatures(self, gameState, action):
        pass

    def getWeights(self, gameState, action):
        pass

    def evaluate(self, gameState, action):
        pass

    def getWidthandHeight(self, gameState):
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        return width, height

    def getWallList(self, gameState):
        return gameState.getWalls().asList()

    def getOpponentList(self, gameState):
        opponentList = []
        for opponentIndex in self.getOpponents(gameState):
            opponentPosition = gameState.getAgentPosition(opponentIndex)
            if not gameState.getAgentState(opponentIndex).isPacman and opponentPosition is not None:
                opponentList.append(opponentPosition)
        return opponentList

    def getOpponentPacmanList(self, gameState):
        opponentPacmanList = []
        for opponentIndex in self.getOpponents(gameState):
            opponentPacmanPosition = gameState.getAgentPosition(opponentIndex)
            if gameState.getAgentState(opponentIndex).isPacman and opponentPacmanPosition is not None:
                opponentPacmanList.append(opponentPacmanPosition)
        return opponentPacmanList

    def areGhostsAround(self, gameState, testCoordinate, inclusiveRangeThreshold):
        """
        Check whether there are ghosts around me within certain distance
        """
        surroundingOpponentList = []
        considerGhostAsSurroundingThreshold = sys.maxsize
        for opponentIndex in self.getOpponents(gameState):
            opponentPosition = gameState.getAgentPosition(opponentIndex)
            if opponentPosition is not None:
                if testCoordinate[0] - inclusiveRangeThreshold <= opponentPosition[0] <= testCoordinate[0] + inclusiveRangeThreshold and testCoordinate[1] - inclusiveRangeThreshold <= opponentPosition[1] <= testCoordinate[1] + inclusiveRangeThreshold:
                    if self.getMazeDistance(testCoordinate, opponentPosition) <= considerGhostAsSurroundingThreshold:
                        surroundingOpponentList.append(opponentPosition)
        return surroundingOpponentList

    def getFoodList(self, gameState):
        return self.getFood(gameState).asList()

    def getCapsuleList(self, gameState):
        capsuleList = self.getCapsules(gameState)
        notNoneCapsuleList = []
        for capsule in capsuleList:
            if capsule is not None:
                notNoneCapsuleList.append(capsule)
        return notNoneCapsuleList

    def getLegalActionOfPosition(self, coordinate, gameState):
        legalAction = []
        x = coordinate[0]
        y = coordinate[1]
        wallList = gameState.getWalls().asList()
        opponentList = []
        for opponentIndex in self.getOpponents(gameState):
            opponentPosition = gameState.getAgentPosition(opponentIndex)
            if not gameState.getAgentState(opponentIndex).isPacman and opponentPosition is not None:
                opponentList.append(opponentPosition)
        if (x, y + 1) not in wallList and (x, y + 1) not in opponentList:
            legalAction.append(Directions.NORTH)
        if (x, y - 1) not in wallList and (x, y - 1) not in opponentList:
            legalAction.append(Directions.SOUTH)
        if (x - 1, y) not in wallList and (x - 1, y) not in opponentList:
            legalAction.append(Directions.WEST)
        if (x + 1, y) not in wallList and (x + 1, y) not in opponentList:
            legalAction.append(Directions.EAST)
        if (x, y + 1) not in opponentList and (x, y - 1) not in opponentList and (x + 1, y) not in opponentList and (x - 1, y) not in opponentList:
            legalAction.append(Directions.STOP)
        return legalAction

    def closestObject(self, listOfObjects, gameState):
        currentPosition = gameState.getAgentPosition(self.index)
        closestObj = None
        closestDistance = sys.maxsize
        for candidateObject in listOfObjects:
            distance = self.getMazeDistance(candidateObject, currentPosition)
            if distance < closestDistance:
                closestDistance = distance
                closestObj = candidateObject
        return closestObj, closestDistance

    def closestObjectUsingPosition(self, listOfObjects, currentPosition):
        closestObj = None
        closestDistance = sys.maxsize
        for candidateObject in listOfObjects:
            distance = self.getMazeDistance(candidateObject, currentPosition)
            if distance < closestDistance:
                closestDistance = distance
                closestObj = candidateObject
        return closestObj, closestDistance

    def farthestObjectUsingPosition(self, listOfObjects, currentPosition):
        farthestObj = None
        farthestDistance = -1 * sys.maxsize
        for candidateObject in listOfObjects:
            distance = self.getMazeDistance(candidateObject, currentPosition)
            if distance > farthestDistance:
                farthestObj = candidateObject
                farthestDistance = distance
        return farthestObj, farthestDistance

    def bestPortalY(self, wallList, isRed, boardWidth, boardHeight):
        candidates = []
        if isRed:
            for y in range(1, boardHeight):
                if (int(boardWidth / 2) - 1, y) not in wallList:
                    candidates.append((int(boardWidth / 2) - 1, y))
        else:
            for y in range(1, boardHeight):
                if (int(boardWidth / 2), y) not in wallList:
                    candidates.append((int(boardWidth / 2), y))
        return candidates[int(len(candidates) / 2)]

    def breadthFirstSearch(self, problem, avoidCoordinate, isRed, boardWidth, wallList, opponentList):
        """
        For testing only, currently not used.
        """
        for element in opponentList:
            if element not in wallList:
                wallList.append(element)
        open = util.Queue()
        init = (problem.getStartState(), [Directions.STOP], 0)
        open.push(init)
        closed = []
        while not open.isEmpty():
            currNode = open.pop()
            currState = currNode[0]
            currPath = currNode[1]
            currCost = currNode[2]
            if (isRed and currState[0] < int(boardWidth / 2)) or (not isRed and currState[0] >= int(boardWidth / 2)):
                return currPath[1:]
            else:
                if currState not in closed:
                    closed.append(currState)
                    successors = problem.getSuccessors(currState)
                    if len(successors) > 0:
                        for each in successors:
                            if each[0] not in closed and each[0] != avoidCoordinate and each[0] not in wallList:
                                temp = (each[0], currPath + [each[1]], currCost + each[2])
                                open.push(temp)
        return []

    def depthFirstSearch(self, problem, avoidCoordinate, isRed, boardWidth, wallList, opponentList):
        """
        Used to define safeCoordinates
        """
        for element in opponentList:
            if element not in wallList:
                wallList.append(element)
        open = util.Stack()
        initState = (problem.getStartState(), [Directions.STOP], 0)
        open.push(initState)
        closed = []
        while not open.isEmpty():
            currState = open.pop()
            currPos = currState[0]
            currPath = currState[1]
            currCost = currState[2]
            if (isRed and currPos[0] < int(boardWidth / 2)) or (not isRed and currPos[0] >= int(boardWidth / 2)):
                return currPath[1:]
            else:
                closed.append(currPos)
            if currState not in closed:
                successors = problem.getSuccessors(currPos)
                if len(successors) > 0:
                    for each in successors:
                        if each[0] not in closed and each[0] != avoidCoordinate and each[0] not in wallList:
                            temp = (each[0], currPath + [each[1]], currCost + each[2])
                            open.push(temp)
        return []

    def depthFirstSearchSafeDetector(self, problem, currentCoordinate, wallList, opponentList):
        for element in opponentList:
            if element not in wallList:
                wallList.append(element)
        if currentCoordinate not in wallList:
            wallList.append(currentCoordinate)
        open = util.Stack()
        initState = (problem.getStartState(), [Directions.STOP], 0)
        open.push(initState)
        closed = []
        while not open.isEmpty():
            currState = open.pop()
            currPos = currState[0]
            currPath = currState[1]
            currCost = currState[2]
            if currPos in self.safeCoordinates:
                return currPath[1:]
            else:
                closed.append(currPos)
            if currState not in closed:
                successors = problem.getSuccessors(currPos)
                if len(successors) > 0:
                    for each in successors:
                        if each[0] not in closed and each[0] != currentCoordinate and each[0] not in wallList:
                            temp = (each[0], currPath + [each[1]], currCost + each[2])
                            open.push(temp)
        return []




###########################
# Heuristic Search Agents #
###########################

'''
    Heuristic Search A* Agents
    The technique we used here is Heuristic Search. We applied A* algorithm, with pacman specific goal and heuristics.
'''

class WaStarInvader(DummyAgent):

    def chooseLegalRandomAction(self, currentPosition, wallList):
        """
        When heuristic search finds no path, legal random action may be used.
        This function is used very infrequently, under normal situation heuristic search is able to find a path.
        """
        actions = []
        x, y = currentPosition[0], currentPosition[1]
        if (x + 1, y) not in wallList:
            actions.append(Directions.EAST)
        if (x - 1, y) not in wallList:
            actions.append(Directions.WEST)
        if (x, y + 1) not in wallList:
            actions.append(Directions.NORTH)
        if (x, y - 1) not in wallList:
            actions.append(Directions.SOUTH)
        if len(actions) == 0:
            actions.append(Directions.STOP)
        else:
            selectedAction = random.choice(actions)
            actions[0] = selectedAction
        return actions

    def chooseLegalRandomPatrolAction(self, currentPosition, wallList, isRed, boardWidth):
        actions = []
        x, y = currentPosition[0], currentPosition[1]
        if isRed:
            if (x + 1, y) not in wallList and x + 1 < int(boardWidth / 2) and x + 1 >= int(boardWidth / 2) - 2:
                actions.append(Directions.EAST)
            if (x - 1, y) not in wallList and x - 1 < int(boardWidth / 2) and x - 1 >= int(boardWidth / 2) - 2:
                actions.append(Directions.WEST)
            if (x, y + 1) not in wallList and x < int(boardWidth / 2) and x >= int(boardWidth / 2) - 2:
                actions.append(Directions.NORTH)
            if (x, y - 1) not in wallList and x < int(boardWidth / 2) and x >= int(boardWidth / 2) - 2:
                actions.append(Directions.SOUTH)
            if len(actions) == 0:
                actions.append(Directions.STOP)
        else:
            if (x + 1, y) not in wallList and x + 1 >= int(boardWidth / 2) and x + 1 < int(boardWidth / 2) + 2:
                actions.append(Directions.EAST)
            if (x - 1, y) not in wallList and x - 1 >= int(boardWidth / 2) and x - 1 < int(boardWidth / 2) + 2:
                actions.append(Directions.WEST)
            if (x, y + 1) not in wallList and x >= int(boardWidth / 2) and x < int(boardWidth / 2) + 2:
                actions.append(Directions.NORTH)
            if (x, y - 1) not in wallList and x >= int(boardWidth / 2) and x < int(boardWidth / 2) + 2:
                actions.append(Directions.SOUTH)
            if len(actions) == 0:
                actions.append(Directions.STOP)
        selectedAction = random.choice(actions)
        actions[0] = selectedAction
        return actions

    def bestAvoidGhostAction(self, gameState, currentPosition, wallList, opponentList, capsuleList):
        """
        When heuristic search finds no path, this function may be used.
        This function is used very infrequently, under normal situation heuristic search is able to find a path.
        """
        notNoneCapsuleList = []
        for capsule in capsuleList:
            if capsule is not None:
                notNoneCapsuleList.append(capsule)
        if len(notNoneCapsuleList) != 0:
            for capsule in notNoneCapsuleList:
                goCapsuleProblem = PositionSearchProblem(gameState, currentPosition, goal=capsule)
                actions = wastarSearch(goCapsuleProblem, manhattanHeuristic)
                if len(actions) != 0:
                    return actions

        if len(opponentList) != 0:
            x, y = currentPosition[0], currentPosition[1]
            newStartingPoint = []
            if (x + 1, y) not in wallList and (x + 1, y) not in opponentList:
                newStartingPoint.append((x + 1, y))
            if (x - 1, y) not in wallList and (x - 1, y) not in opponentList:
                newStartingPoint.append((x - 1, y))
            if (x, y + 1) not in wallList and (x, y + 1) not in opponentList:
                newStartingPoint.append((x, y + 1))
            if (x, y - 1) not in wallList and (x, y - 1) not in opponentList:
                newStartingPoint.append((x, y - 1))
            safeLeadingDirection = []
            for startingPoint in newStartingPoint:
                if startingPoint in self.safeCoordinates:
                    path = ["placeholder"]
                else:
                    dfsProblem = PositionSearchProblem(gameState, startingPoint)
                    path = self.depthFirstSearchSafeDetector(dfsProblem, currentPosition, self.getWallList(gameState), self.getOpponentList(gameState))
                if len(path) != 0:
                    if startingPoint == (x + 1, y):
                        safeLeadingDirection.append(Directions.EAST)
                    elif startingPoint == (x - 1, y):
                        safeLeadingDirection.append(Directions.WEST)
                    elif startingPoint == (x, y + 1):
                        safeLeadingDirection.append(Directions.NORTH)
                    elif startingPoint == (x, y - 1):
                        safeLeadingDirection.append(Directions.SOUTH)
            if len(safeLeadingDirection) != 0:
                selectedAction = random.choice(safeLeadingDirection)
                safeLeadingDirection[0] = selectedAction
                return safeLeadingDirection

        if len(opponentList) != 0:
            distanceToGhost = -1 * sys.maxsize + 1
            wisestAction = []
            distanceStorage = {}
            x, y = currentPosition[0], currentPosition[1]
            if (x + 1, y) not in wallList:
                tempDistance = 0
                for opponent in opponentList:
                    tempDistance += self.getMazeDistance((x + 1, y), opponent)
                distanceStorage[Directions.EAST] = tempDistance
                if tempDistance > distanceToGhost:
                    distanceToGhost = tempDistance
            if (x - 1, y) not in wallList:
                tempDistance = 0
                for opponent in opponentList:
                    tempDistance += self.getMazeDistance((x - 1, y), opponent)
                distanceStorage[Directions.WEST] = tempDistance
                if tempDistance > distanceToGhost:
                    distanceToGhost = tempDistance
            if (x, y + 1) not in wallList:
                tempDistance = 0
                for opponent in opponentList:
                    tempDistance += self.getMazeDistance((x, y + 1), opponent)
                distanceStorage[Directions.NORTH] = tempDistance
                if tempDistance > distanceToGhost:
                    distanceToGhost = tempDistance
            if (x, y - 1) not in wallList:
                tempDistance = 0
                for opponent in opponentList:
                    tempDistance += self.getMazeDistance((x, y - 1), opponent)
                distanceStorage[Directions.SOUTH] = tempDistance
                if tempDistance > distanceToGhost:
                    distanceToGhost = tempDistance
            for action in distanceStorage.keys():
                if distanceStorage[action] == distanceToGhost:
                    wisestAction.append(action)
            if len(wisestAction) == 0:
                return [self.chooseLegalRandomAction(currentPosition, wallList)[0]]
            return wisestAction
        else:
            return [self.chooseLegalRandomAction(currentPosition, wallList)[0]]

    def updateHungrySteps(self, currentPosition, nextAction, foodList, capsuleList, opponentPacmanList, opponentList):
        """
        NOTE: This function is for testing only, currently not used.
        Maintains the hungry steps
        """
        x, y = currentPosition[0], currentPosition[1]
        if nextAction == Directions.NORTH:
            if (x, y + 1) in foodList or (x, y + 1) in capsuleList or (x, y + 1) in opponentPacmanList or (x, y + 1) in opponentList:
                self.hungrySteps = 0
            else:
                self.hungrySteps += 1
        if nextAction == Directions.SOUTH:
            if (x, y - 1) in foodList or (x, y - 1) in capsuleList or (x, y - 1) in opponentPacmanList or (x, y - 1) in opponentList:
                self.hungrySteps = 0
            else:
                self.hungrySteps += 1
        if nextAction == Directions.WEST:
            if (x - 1, y) in foodList or (x - 1, y) in capsuleList or (x - 1, y) in opponentPacmanList or (x - 1, y) in opponentList:
                self.hungrySteps = 0
            else:
                self.hungrySteps += 1
        if nextAction == Directions.EAST:
            if (x + 1, y) in foodList or (x + 1, y) in capsuleList or (x + 1, y) in opponentPacmanList or (x + 1, y) in opponentList:
                self.hungrySteps = 0
            else:
                self.hungrySteps += 1
        if nextAction == Directions.STOP:
            self.hungrySteps += 1
        return

    def retreat(self, gameState):
        """
        Heuristic Search Retreat Mode
        Used for agent to retreat back home using heuristic search algorithm.
        """
        width, height = self.getWidthandHeight(gameState)
        currentPosition = gameState.getAgentPosition(self.index)
        wallList = gameState.getWalls().asList()
        homeWidth = int(width / 2)
        candidateHomeList = []
        if self.red:
            for i in range(1, height):
                if (homeWidth - 1, i) not in wallList:
                    candidateHomeList.append((homeWidth - 1, i))
        else:
            for i in range(1, height):
                if (homeWidth, i) not in wallList:
                    candidateHomeList.append((homeWidth, i))
        closestHome, distance = self.closestObject(candidateHomeList, gameState)
        goHomeProblem = PositionSearchProblem(gameState, currentPosition, goal = closestHome)
        actions = wastarSearch(goHomeProblem, manhattanHeuristic)
        if len(actions) == 0:
            actions = self.bestAvoidGhostAction(gameState, currentPosition, wallList, self.getOpponentList(gameState), self.getCapsuleList(gameState))
        return actions

    def chooseAction(self, gameState):
        """
        Using Heuristic Search to decide actions for pacman agent
        Under different scenario, different goal and heuristic will be used.
        """
        scaredTime = 0
        opponentDict = dict()
        numberOfScaredGhost = 0
        countAsScaredThreshold = 0
        for opponentIndex in self.getOpponents(gameState):
            if not gameState.getAgentState(opponentIndex).isPacman:
                tempScaredTime = gameState.data.agentStates[opponentIndex].scaredTimer
                opponentDict[(opponentIndex, gameState.getAgentPosition(opponentIndex))] = tempScaredTime
                if tempScaredTime > countAsScaredThreshold:
                    numberOfScaredGhost += 1
                if tempScaredTime > scaredTime:
                    scaredTime = tempScaredTime

        opponentList = self.getOpponentList(gameState)
        currentPosition = gameState.getAgentPosition(self.index)
        closestOpponent, closestOpponentDistance = self.closestObjectUsingPosition(opponentList, currentPosition)

        self.updateMode(gameState, scaredTime, closestOpponentDistance)

        if self.debug_message: print("============INVADER============")
        if self.debug_message: print("Mode: " + self.mode)
        if self.debug_message: print("Position: " + str(gameState.getAgentPosition(self.index)))

        foodList = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)
        opponentPacmanList = self.getOpponentPacmanList(gameState)
        if self.debug_message: print("Capsule List: " + str(capsuleList))
        if self.debug_message: print("Opponent List: " + str(opponentList))
        if self.debug_message: print("Opponent Pacman List: " + str(opponentPacmanList))
        if closestOpponent is not None:
            if self.debug_message: print("Closest Opponent Ghost: " + str(closestOpponent) + ": " + str(closestOpponentDistance))

        """
        Handle Walls
        """
        wallList = self.getWallList(gameState)
        original_wall_grids = gameState.data.layout.walls
        updatedGameState = gameState.deepCopy()
        grid_width, grid_height = self.getWidthandHeight(gameState)
        wall_grids = gameState.getWalls().data

        if len(opponentList) != 0 and numberOfScaredGhost != len(list(opponentDict.keys())):
            if self.debug_message: print("Number of Opponent Around Me: " + str(len(opponentList)))
            if self.debug_message: print("Number of Scared Opponent Around Me: " + str(numberOfScaredGhost))
            for candidateOpponent in opponentList:
                for key in opponentDict:
                    if candidateOpponent == key[1] and opponentDict[key] <= countAsScaredThreshold:
                        distance = self.getMazeDistance(candidateOpponent, currentPosition)
                        if self.debug_message: print("Not Scared Opponent " + str(candidateOpponent) + " Distance: " + str(distance))
                        if candidateOpponent not in wallList:
                            wallList.append(candidateOpponent)
                            wall_grids[candidateOpponent[0]][candidateOpponent[1]] = True
                        considerGhostSurroundingAreaDistanceThreshold = 3
                        if distance <= considerGhostSurroundingAreaDistanceThreshold:
                            x, y = candidateOpponent[0], candidateOpponent[1]
                            if (x + 1, y) not in wallList:
                                wallList.append((x + 1, y))
                                wall_grids[x + 1][y] = True
                            if (x - 1, y) not in wallList:
                                wallList.append((x - 1, y))
                                wall_grids[x - 1][y] = True
                            if (x, y + 1) not in wallList:
                                wallList.append((x, y + 1))
                                wall_grids[x][y + 1] = True
                            if (x, y - 1) not in wallList:
                                wallList.append((x, y - 1))
                                wall_grids[x][y - 1] = True

        boardWidth, boardHeight = self.getWidthandHeight(updatedGameState)
        if self.red and currentPosition[0] == int(boardWidth / 2) - 1:
            if (int(boardWidth / 2), currentPosition[1]) in opponentList:
                for key in opponentDict:
                    if (int(boardWidth / 2), currentPosition[1]) == key[1] and opponentDict[key] <= countAsScaredThreshold:
                        wallList.append((int(boardWidth / 2), currentPosition[1]))
                        wall_grids[int(boardWidth / 2)][currentPosition[1]] = True
        if self.red and currentPosition[0] == int(boardWidth / 2):
            if (int(boardWidth / 2) - 1, currentPosition[1]) in opponentPacmanList:
                wallList.append((int(boardWidth / 2) - 1, currentPosition[1]))
                wall_grids[(int(boardWidth / 2) - 1)][currentPosition[1]] = True
        if not self.red and currentPosition[0] == int(boardWidth / 2):
            if (int(boardWidth / 2) - 1, currentPosition[1]) in opponentList:
                for key in opponentDict:
                    if (int(boardWidth / 2) - 1, currentPosition[1]) == key[1] and opponentDict[key] <= countAsScaredThreshold:
                        wallList.append((int(boardWidth / 2) - 1, currentPosition[1]))
                        wall_grids[(int(boardWidth / 2) - 1)][currentPosition[1]] = True
        if not self.red and currentPosition[0] == int(boardWidth / 2) - 1:
            if (int(boardWidth / 2), currentPosition[1]) in opponentPacmanList:
                wallList.append((int(boardWidth / 2), currentPosition[1]))
                wall_grids[int(boardWidth / 2)][currentPosition[1]] = True

        myScaredTime = 0
        if not updatedGameState.getAgentState(self.index).isPacman:
            myScaredTime = updatedGameState.data.agentStates[self.index].scaredTimer

        if len(opponentPacmanList) != 0 and self.mode == "invader home mode" and myScaredTime > 0:
            if self.debug_message: print("My Scared Timer: " + str(myScaredTime))
            for candidateOpponentPacman in opponentPacmanList:
                distance = self.getMazeDistance(candidateOpponentPacman, currentPosition)
                if self.debug_message: print("Opponent Pacman " + str(candidateOpponentPacman) + " Distance: " + str(distance))
                if candidateOpponentPacman not in wallList:
                    wallList.append(candidateOpponentPacman)
                    wall_grids[candidateOpponentPacman[0]][candidateOpponentPacman[1]] = True
                considerPacmanSurroundingAreaDistanceThreshold = 2
                if distance <= considerPacmanSurroundingAreaDistanceThreshold:
                    x, y = candidateOpponentPacman[0], candidateOpponentPacman[1]
                    if (x + 1, y) not in wallList:
                        wallList.append((x + 1, y))
                        wall_grids[x + 1][y] = True
                    if (x - 1, y) not in wallList:
                        wallList.append((x - 1, y))
                        wall_grids[x - 1][y] = True
                    if (x, y + 1) not in wallList:
                        wallList.append((x, y + 1))
                        wall_grids[x][y + 1] = True
                    if (x, y - 1) not in wallList:
                        wallList.append((x, y - 1))
                        wall_grids[x][y - 1] = True

        if updatedGameState.getAgentState(self.index).isPacman:
            homeBorderThreshold = 3
            boardWidth, boardHeight = self.getWidthandHeight(updatedGameState)
            if self.red and int(boardWidth / 2) - homeBorderThreshold >= 0:
                newBorderX = int(boardWidth / 2) - homeBorderThreshold
                for h in range(boardHeight):
                    if (newBorderX, h) not in wallList:
                        wallList.append((newBorderX, h))
                        wall_grids[newBorderX][h] = True
            elif not self.red and int(boardWidth / 2) - 1 + homeBorderThreshold < boardWidth:
                newBorderX = int(boardWidth / 2) - 1 + homeBorderThreshold
                for h in range(boardHeight):
                    if (newBorderX, h) not in wallList:
                        wallList.append((newBorderX, h))
                        wall_grids[newBorderX][h] = True

        combined_wall_grid = game.Grid(grid_width, grid_height, False)
        for i in range(grid_width):
            for j in range(grid_height):
                combined_wall_grid.data[i][j] = wall_grids[i][j]
        updatedGameState.data.layout.walls = combined_wall_grid

        """
        The heuristic search will be done below.
        During the heuristic search, under different circumstances, different goal and different heuristic will be used.
        Each mode stands for different scenarios.
        """
        if self.mode == "invader home mode":
            """
            Goal Selection
            """
            goodList = []
            huntingPacmanScoreThreshold = 7
            score = self.getScore(updatedGameState)
            if score >= huntingPacmanScoreThreshold and myScaredTime == 0:
                if self.debug_message: print("Score: " + str(score))
                if self.debug_message: print("Nearby Pacman List: " + str(opponentPacmanList))
                for element in opponentPacmanList:
                    if element not in goodList:
                        goodList.append(element)
            if len(goodList) == 0:
                if len(opponentList) == 0:
                    for element in capsuleList:
                        if element not in goodList:
                            goodList.append(element)
                    for element in foodList:
                        if element not in goodList:
                            goodList.append(element)
                else:
                    if len(goodList) == 0:
                        for element in capsuleList:
                            if element not in goodList:
                                goodList.append(element)
                    if len(goodList) == 0:
                        for element in foodList:
                            if element in self.safeCoordinates and element not in goodList:
                                goodList.append(element)
                    if len(goodList) == 0:
                        for element in foodList:
                            if element not in goodList:
                                goodList.append(element)

            if updatedGameState.data.timeleft <= 180:
                goodList = []
                if myScaredTime == 0:
                    if len(opponentPacmanList) != 0:
                        for element in opponentPacmanList:
                            if element not in goodList:
                                goodList.append(element)

            if len(goodList) == 0:
                if (self.red and currentPosition[0] >= int(boardWidth / 2) - 2) or (not self.red and currentPosition[0] < int(boardWidth / 2) + 2):
                    actions = self.chooseLegalRandomPatrolAction(currentPosition, wallList, self.red, boardWidth)
                    return actions[0]
                else:
                    goodList.append(self.bestPortalY(wallList, self.red, boardWidth, boardHeight))

            """
            Heuristic Search
            """
            closestGood, distance = self.closestObject(goodList, updatedGameState)
            if self.debug_message: print("Goal: " + str(closestGood))
            if abs(closestGood[0] - currentPosition[0]) > 10 and ((self.red and currentPosition[0] <= int(boardWidth / 4)) or (not self.red and currentPosition[0] >= int(boardWidth / 4 * 3))):
                actions = self.pathDict[currentPosition][closestGood].copy()
                allowedActions = self.getLegalActionOfPosition(currentPosition, updatedGameState)
                if actions[0] == Directions.STOP:
                    actions[0] = actions[1]
                if actions[0] not in allowedActions:
                    selectedAction = random.choice(allowedActions)
                    actions[0] = selectedAction
            else:
                closestGoodProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestGood)
                actions = wastarSearch(closestGoodProblem, manhattanHeuristic)
            if len(actions) == 0:
                actions = self.chooseLegalRandomAction(currentPosition, wallList)
            if self.debug_message: print("Action: " + actions[0])
            if self.debug_message: print("===============================")
            if self.debug_message: print()
            if self.debug_message: print()
            updatedGameState.data.layout.walls = original_wall_grids
            gameState = updatedGameState
            return actions[0]


        elif self.mode == "invader hunting mode" and len(opponentList) == 0:
            """
            Goal Selection
            """
            if updatedGameState.getAgentState(self.index).numCarrying == 0:
                foodDistanceThreshold = sys.maxsize
                capsuleDistanceThreshold = sys.maxsize
                pacmanDistanceThreshold = sys.maxsize
            else:
                foodDistanceThreshold = 6
                capsuleDistanceThreshold = 6
                pacmanDistanceThreshold = 6

            goodList = []
            closestFood = None
            closestDistance = sys.maxsize
            for element in foodList:
                tempDistance = self.getMazeDistance(currentPosition, element)
                if element not in goodList and tempDistance < foodDistanceThreshold:
                    goodList.append(element)
                    if tempDistance < closestDistance:
                        closestDistance = tempDistance
                        closestFood = element
            for element in capsuleList:
                tempDistance = self.getMazeDistance(currentPosition, element)
                if element not in goodList and tempDistance < capsuleDistanceThreshold:
                    goodList.append(element)
                    if tempDistance < closestDistance:
                        closestDistance = tempDistance
                        closestFood = element

            huntingPacmanScoreThreshold = 3
            score = self.getScore(updatedGameState)
            """
            if score >= huntingPacmanScoreThreshold:
                for element in opponentPacmanList:
                    tempDistance = self.getMazeDistance(currentPosition, element)
                    if element not in goodList and tempDistance < pacmanDistanceThreshold:
                        goodList.append(element)
                        if tempDistance < closestDistance:
                            closestDistance = tempDistance
                            closestFood = element
            """

            """
            Heuristic Search
            """
            if len(goodList) == 0 or closestFood is None:
                if self.debug_message: print("Mode Changes: invader retreat mode")
                actions = self.retreat(updatedGameState)
                updatedGameState.data.layout.walls = original_wall_grids
                gameState = updatedGameState
                return actions[0]

            closestFoodProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestFood)
            if self.debug_message: print("Goal: " + str(closestFood))
            actions = wastarSearch(closestFoodProblem, manhattanHeuristic)

            if closestFood in foodList:
                if self.debug_message: print("Goal Type: Food")
            elif closestFood in capsuleList:
                if self.debug_message: print("Goal Type: Capsule")
            else:
                if self.debug_message: print("Goal Type: Opponent Pacman At Home")
                if self.debug_message: print("Opponent Pacman List: " + str(opponentPacmanList))
                if self.debug_message: print("Score: " + str(score))
            if len(actions) == 0:
                actions = self.chooseLegalRandomAction(currentPosition, wallList)
            if self.debug_message: print("Action: " + actions[0])
            if self.debug_message: print("===============================")
            if self.debug_message: print()
            if self.debug_message: print()
            updatedGameState.data.layout.walls = original_wall_grids
            gameState = updatedGameState
            return actions[0]

        elif self.mode == "invader hunting mode" and len(opponentList) != 0:
            """
            Goal Selection
            """
            if self.debug_message: print("Opponent List: " + str(opponentList))
            safeList = []
            capsulePrioritized = False
            if len(capsuleList) != 0:
                capsulePrioritizedDistanceThreshold = 6
                for capsule in capsuleList:
                    if self.getMazeDistance(currentPosition, capsule) <= capsulePrioritizedDistanceThreshold:
                        capsulePrioritized = True
            if not capsulePrioritized:
                score = self.getScore(updatedGameState)
                foodCarrying = updatedGameState.getAgentState(self.index).numCarrying
                if len(opponentList) == 2:
                    foodCarryingReturnThreshold = 6
                else:
                    foodCarryingReturnThreshold = 7
                if score < 3 * foodCarryingReturnThreshold:
                    if foodCarrying >= foodCarryingReturnThreshold:
                        actions = self.retreat(updatedGameState)
                        updatedGameState.data.layout.walls = original_wall_grids
                        gameState = updatedGameState
                        return actions[0]
            if not capsulePrioritized:
                for foodCoordinate in foodList:
                    if len(self.areGhostsAround(updatedGameState, foodCoordinate, 4)) > 1:
                        if self.isSafeCoordinate(foodCoordinate, updatedGameState):
                            safeList.append(foodCoordinate)
                    elif foodCoordinate in self.safeCoordinates:
                        safeList.append(foodCoordinate)

            for capsuleCoordinate in capsuleList:
                safeList.append(capsuleCoordinate)

            if len(safeList) == 0:
                actions = self.retreat(updatedGameState)
                updatedGameState.data.layout.walls = original_wall_grids
                gameState = updatedGameState
                return actions[0]

            """
            Heuristic Search
            """
            closestSafe, distance = self.closestObject(safeList, updatedGameState)
            if self.debug_message: print("Goal: " + str(closestSafe))
            if closestSafe is not None:
                closestFoodProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestSafe)
                actions = wastarSearch(closestFoodProblem, manhattanHeuristic)
            else:
                actions = self.retreat(updatedGameState)
                if self.debug_message: print("Go Home")
                if len(actions) == 0:
                    actions = self.bestAvoidGhostAction(updatedGameState, currentPosition, wallList, self.getOpponentList(updatedGameState), self.getCapsuleList(updatedGameState))
                if self.debug_message: print("Action: " + str(actions[0]))
                return actions[0]

            if closestSafe in foodList:
                if self.debug_message: print("Goal Type: Safe Food")
            elif closestSafe in capsuleList:
                if self.debug_message: print("Goal Type: Safe Capsule")
            else:
                if self.debug_message: print("Goal Type: Closest Home")
            if len(actions) == 0:
                actions = self.retreat(updatedGameState)
                if len(actions) == 0:
                    actions = self.bestAvoidGhostAction(updatedGameState, currentPosition, wallList, self.getOpponentList(updatedGameState), self.getCapsuleList(updatedGameState))
            if self.debug_message: print("Action: " + actions[0])
            if self.debug_message: print("===============================")
            if self.debug_message: print()
            if self.debug_message: print()
            updatedGameState.data.layout.walls = original_wall_grids
            gameState = updatedGameState
            return actions[0]

        elif self.mode == "invader power mode":
            """
            Goal Selection
            """
            goodList = []
            for element in foodList:
                if element not in goodList:
                    goodList.append(element)
            for element in capsuleList:
                if element not in goodList and element not in wallList:
                    goodList.append(element)
            scaredTimerCountingDownThreshold = 8
            scaredOpponentGhostDistance = 1
            for element in opponentList:
                for key in opponentDict:
                    if element == key[1] and element not in wallList and opponentDict[key] >= scaredTimerCountingDownThreshold and element not in goodList:
                        if self.getMazeDistance(currentPosition, element) <= scaredOpponentGhostDistance:
                            goodList.append(element)

            """
            Heuristic Search
            """
            if len(goodList) == 0:
                actions = self.retreat(updatedGameState)
                if self.debug_message: print("Goal Type: Closest Home")
                if len(actions) == 0:
                    actions = self.bestAvoidGhostAction(updatedGameState, currentPosition, wallList, self.getOpponentList(updatedGameState), self.getCapsuleList(updatedGameState))
                if self.debug_message: print("Action: " + actions[0])
                if self.debug_message: print("===============================")
                if self.debug_message: print()
                if self.debug_message: print()
                updatedGameState.data.layout.walls = original_wall_grids
                gameState = updatedGameState
                return actions[0]

            closestFood, distance = self.closestObject(goodList, updatedGameState)
            if self.debug_message: print("Goal: " + str(closestFood))
            closestFoodProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestFood)
            actions = wastarSearch(closestFoodProblem, manhattanHeuristic)
            if closestFood in capsuleList:
                if self.debug_message: print("Goal Type: Capsule")
            elif closestFood in opponentList:
                if self.debug_message: print("Goal Type: Opponent Ghost")
            else:
                if self.debug_message: print("Goal Type: Food")
            if len(actions) == 0:
                actions = self.retreat(updatedGameState)
                if len(actions) == 0:
                    actions = self.bestAvoidGhostAction(updatedGameState, currentPosition, wallList, self.getOpponentList(updatedGameState), self.getCapsuleList(updatedGameState))
            if self.debug_message: print("Action: " + actions[0])
            if self.debug_message: print("Capsule Eaten: " + str(updatedGameState.data._capsuleEaten))
            if self.debug_message: print("Scared Timer: " + str(scaredTime))
            if self.debug_message: print("===============================")
            if self.debug_message: print()
            if self.debug_message: print()
            updatedGameState.data.layout.walls = original_wall_grids
            gameState = updatedGameState
            return actions[0]


        elif self.mode == "invader retreat mode":
            """
            Goal Selection & Heuristic Search
            """
            actions = self.retreat(updatedGameState)
            if len(actions) == 0:
                actions = self.bestAvoidGhostAction(updatedGameState, currentPosition, wallList, self.getOpponentList(updatedGameState), self.getCapsuleList(updatedGameState))
            updatedGameState.data.layout.walls = original_wall_grids
            gameState = updatedGameState
            return actions[0]

    def updateScore(self, gameState):
        gameState.data.score = len(gameState.getRedFood().asList()) - len(gameState.getBlueFood().asList())

    def updateMode(self, gameState, scaredTime, closestOpponentDistance):
        boardWidth, boardHeight = self.getWidthandHeight(gameState)
        currentPosition = gameState.getAgentPosition(self.index)
        timeLeft = gameState.data.timeleft
        if timeLeft <= 180 and ((self.red and currentPosition[0] >= int(boardWidth / 2)) or (not self.red and currentPosition[0] < int(boardWidth / 2))):
            self.mode = "invader retreat mode"
            return
        if len(self.getFood(gameState).asList()) <= 2:
            self.mode = "invader retreat mode"
            return
        if self.red and currentPosition[0] < int(boardWidth / 2):
            self.mode = "invader home mode"
            return
        elif not self.red and currentPosition[0] >= int(boardWidth / 2):
            self.mode = "invader home mode"
            return
        else:
            if gameState.data._capsuleEaten is not None:
                self.mode = "invader power mode"
                return
            scaredTimerCountingDownThreshold = 8
            if gameState.data._capsuleEaten is None and scaredTime >= scaredTimerCountingDownThreshold:
                self.mode = "invader power mode"
                return
            if gameState.data._capsuleEaten is None and scaredTime < scaredTimerCountingDownThreshold:
                self.mode = "invader hunting mode"
            self.mode = "invader hunting mode"

    def evaluate(self, gameState, action):
        pass

    def getFeatures(self, gameState, action):
        pass

    def getWeights(self, gameState, action):
        pass



#################################
#   WASTAR DEFENDER
#################################

class WaStarDefender(DummyAgent):
    """
      Defender Behavior design:
      1. Patrol around home area -- search and chase pacman (priority: High)
      2.

    """

    def __init__(self, index):
        super().__init__(index)

        # Layout related
        self.maze_dim = None
        self.boarder_mid = None

        # game state variables
        self.initialPosition = None
        self.currentPosition = None
        self.opponentIndices = None
        self.searching = None
        self.eatenFoods = None
        self.walls = None
        self.eatDefender = None
        self.my_zone = None
        self.potentialInvaders = None
        self.teamInvader = None
        self.mode = None
        self.target = None
        self.foodEaten = None
        self.initialEnemyPosition = None

    def registerInitialState(self, gameState):
        """
        This func will be called when:
            > the first time the chosen agent obj was created
            > at the beginning of each new game
        :param gameState: the initial game state
        """
        CaptureAgent.registerInitialState(self, gameState)
        self.walls = gameState.getWalls()
        # dimensions of the grid world w * h
        self.maze_dim = (gameState.data.layout.width, gameState.data.layout.height)

        self.opponentIndices = self.getOpponents(gameState)

        self.start = gameState.getAgentPosition(self.index)
        self.initialPosition = gameState.getAgentPosition(self.index)
        self.initialEnemyPosition = (self.maze_dim[0] - self.initialPosition[0] - 1, self.maze_dim[1] - self.initialPosition[1] - 1)
        half_width = self.maze_dim[0] // 2
        # If the initial position is on the right side
        if self.initialPosition[0] > half_width:
            self.boarder_mid = (half_width, self.maze_dim[1] // 2)
            self.opponentborder = half_width - 1
            self.my_zone = range(half_width, self.maze_dim[0])
            self.nearHome = range(self.maze_dim[0]-5,self.maze_dim[0])
        else:
            self.boarder_mid = (half_width - 1, self.maze_dim[1] // 2)
            self.opponentborder = half_width
            self.my_zone = range(half_width)
            self.nearHome = range(5)
        while self.walls[self.boarder_mid[0]][self.boarder_mid[1]]:
            self.boarder_mid = (self.boarder_mid[0], self.boarder_mid[1] + 1)

        self.eatDefender = False
        self.myFood = self.getFoodYouAreDefending(gameState).asList()
        self.potentialInvaders = []
        self.boardHeight = gameState.data.layout.height
        for index in self.getTeam(gameState):
            if index != self.index:
                self.teamInvader = index
        self.boardWidth = self.maze_dim[0]
        for x in range(self.boardWidth):
            for y in range(self.boardHeight):
                if self.isSafeCoordinate((x, y), gameState):
                    self.safeCoordinates.append((x, y))
                else:
                    self.riskyCoordinates.append((x, y))

        self.pathDict = self.scanMaze()
        self.mode = None
        self.target = None
        self.foodEaten = None

    def chooseAction(self, gameState):
        """
        The func as the agent's turn commences, choose action accordingly based on
        what mode hs is currently in：
           > As a normal ghost, patrol around by default
             Actions to perform:
                > hunt(), if any enemy pacman detected
                > defend(), if no enemy pacman found

           > As a scared ghost:
                > suicide and expects re-spawning asap, if scared recently
                > flee, if scared for some time
        """
        # update current status
        self.currentPosition = gameState.getAgentPosition(self.index)

        invaders = self.searchInvadersPosition(gameState)
        if invaders:
            self.searching = False
            if gameState.getAgentState(self.index).scaredTimer <= 0:  # As a normal ghost
                return self.hunt(gameState, invaders)
            else:  # As a scared ghost
                return self.scaredAction(gameState, invaders)
        else:
            return self.defend(gameState)  # routine patrol

    def searchInvadersPosition(self, gameState):
        """
        Search all observable invaders (enemy pacmans)
        :param gameState:
        :return: a list of invader position if any, or an empty list
        """
        invaders = []
        for opponent in self.opponentIndices:
            # if in observable area
            if gameState.getAgentPosition(opponent):
                # if the opponent invades
                if gameState.getAgentState(opponent).isPacman:
                    invaders.append(gameState.getAgentPosition(opponent))
                    if opponent not in self.potentialInvaders:
                        self.potentialInvaders.append(opponent)
        return invaders

    def scaredAction(self, gameState, invaders):
        """
         Actions a scared ghost may perform.
        :param gameState:
        :param invaders: a list of invader position if any, or an empty list
        """
        # Pick nearest enemy
        enemyPosition = sorted(invaders, key=lambda pos: pow(pos[0] - self.currentPosition[0], 2) + pow(
            pos[1] - self.currentPosition[1], 2))[0]
        d = self.getMazeDistance(self.initialPosition, self.currentPosition)
        if d < gameState.getAgentState(self.index).scaredTimer - 2:
            return self.suicide(gameState, enemyPosition)
        else:
            return self.flee(gameState, enemyPosition)

    def suicide(self, gameState, enemyPosition):
        print("mode: suicide")
        print(self.currentPosition)
        goHomeProblem = PositionSearchProblem(gameState, self.currentPosition, goal=enemyPosition)
        actions = wastarSearch(goHomeProblem, manhattanHeuristic)
        self.mode = "suicide"
        if len(actions) > 0:
            return actions[0]
        else:
            return 'Stop'

    def defend(self, gameState):
        currentFood = self.getFoodYouAreDefending(gameState).asList()
        if len(self.myFood) > len(currentFood) or self.searching:
            self.searching = True
            eatenFoods = [item for item in self.myFood if item not in currentFood]
            if len(eatenFoods) == 0:
                eatenFoods = self.eatenFoods
            else:
                self.eatenFoods = eatenFoods
            self.myFood = currentFood
            return self.findInvader(gameState, eatenFoods[0])
        else:
            # Eat The Scared Pacman If It is Just Beside Uou
            #
            for opponent in self.opponentIndices:
                op = gameState.getAgentPosition(opponent)
                if op and (not gameState.getAgentState(opponent).isPacman) and gameState.getAgentState(opponent).scaredTimer > 0:
                    if op[1] == self.currentPosition[1] and abs(op[0]-self.currentPosition[0]) == 1:
                        if gameState.getAgentState(opponent).scaredTimer < 10 or self.getMazeDistance(gameState.getAgentPosition(self.teamInvader), self.initialEnemyPosition) > 10:
                            FoodProblem = PositionSearchProblem(gameState, self.currentPosition,goal=op)
                            actions = wastarSearch(FoodProblem, manhattanHeuristic)
                            print("mode: Eat The Unlucky Ghost")
                            print(self.currentPosition)
                            if len(actions) > 0:
                                return actions[0]
                            else:
                                return 'Stop'
            #
            #
            self.myFood = currentFood
            invaders = []
            for opponent in self.potentialInvaders:
                # if in observable area
                if gameState.getAgentPosition(opponent):
                    # if the opponent invades
                    if not gameState.getAgentState(opponent).isPacman:
                        invaders.append(gameState.getAgentPosition(opponent))
            # re = False
            # ops = []
            # for opponent in self.opponentIndices:
            #     # if in observable area
            #     if gameState.getAgentPosition(opponent) and (not gameState.getAgentState(opponent).isPacman) and \
            #             gameState.getAgentState(opponent).scaredTimer <= 0:
            #         ops.append(opponent)
            #         re = True
            if len(invaders) > 0 and (not gameState.getAgentState(self.index).isPacman):
                return self.track(invaders[0], gameState)
            if self.eatDefender:
                return self.stealFood(gameState)

            someoneInvading = False
            for opponent in self.opponentIndices:
                if gameState.getAgentState(opponent).isPacman:
                    someoneInvading = True
                    break
            if someoneInvading:
                if self.mode == "hunt" or self.mode == "findInvader":
                    if self.mode == "hunt":
                        defendFoodProblem = PositionSearchProblem(gameState, self.currentPosition, goal=self.target)
                        d = self.getMazeDistance(self.currentPosition, self.target)
                    if self.mode == "findInvader":
                        defendFoodProblem = PositionSearchProblem(gameState, self.currentPosition, goal=self.foodEaten)
                        d = self.getMazeDistance(self.currentPosition, self.foodEaten)
                    if d >= 12:
                        actions = wastarSearch(defendFoodProblem, manhattanHeuristic)
                    else:
                        actions = wastarSearch(defendFoodProblem, mazeDistanceHeuristic)
                    print("mode: someone invading")
                    print(self.currentPosition)
                    if len(actions) > 0:
                        return actions[0]
                    else:
                        return 'Stop'

            re = False
            for opponent in self.opponentIndices:
                # if in observable area,
                if gameState.getAgentPosition(opponent) and (not gameState.getAgentState(opponent).isPacman) and \
                        gameState.getAgentState(opponent).scaredTimer <= 0:
                    avoid = gameState.getAgentPosition(opponent)
                    re = True
                    break
            if re:
                defendFoodProblem = AvoidProblem(gameState, self.currentPosition, opponent=avoid, goal=self.boarder_mid, myzone=self.my_zone)
                actions = wastarSearch(defendFoodProblem, manhattanHeuristic)
            else:
                if self.currentPosition[0] in self.nearHome:
                    actions = self.pathDict[self.currentPosition][self.boarder_mid].copy()
                else:
                    defendFoodProblem = PositionSearchProblem(gameState, self.currentPosition, goal=self.boarder_mid)
                    actions = wastarSearch(defendFoodProblem, manhattanHeuristic)
            self.mode = "routine patrol"
            if len(actions) > 0:
                return actions[0]
            else:
                return 'Stop'

    def track(self, invader, gameState, other = None):
        trackProblem = TrackProblem(gameState, self.currentPosition, opponentborder=self.opponentborder, goal=invader, self_border=self.boarder_mid[0], height=self.boardHeight,myZone=self.my_zone, other = other)
        actions = wastarSearch(trackProblem, manhattanHeuristic)
        self.mode = "track"
        print("mode: track")
        print(self.currentPosition)
        if len(actions) > 0:
            return actions[0]
        else:
            return 'Stop'

    def stealFood(self, gameState):
        foods = self.getFood(gameState).asList()
        capsules = self.getCapsules(gameState)
        d = 6
        d_relax = 10
        d_relax_c = 15
        temp = []
        absolutely_safe = True
        for opponent in self.opponentIndices:
            if gameState.getAgentPosition(opponent) and gameState.getAgentState(opponent).scaredTimer <= 10:
                absolutely_safe = False
                break
        # Very safe, be ambitious!
        if absolutely_safe:
            # Help invader, Eat the capsule
            if gameState.getAgentState(self.teamInvader).isPacman:
                for capsule in capsules:
                    ds = []
                    for i in range(self.maze_dim[1]):
                        if (self.boarder_mid[0], i) not in self.getWallList(gameState):
                            t = self.getMazeDistance(capsule, (self.boarder_mid[0], i))
                            ds.append(t)
                    minds = min(ds)
                    if minds < 12:
                        temp.append((capsule, True))
            # Don't Eat Capsule
            # Food: [d < 8 or (d < 12 and safe)]
            for food in foods:
                ds = []
                for i in range(self.maze_dim[1]):
                    if (self.boarder_mid[0], i) not in self.getWallList(gameState):
                        t = self.getMazeDistance(food, (self.boarder_mid[0], i))
                        ds.append(t)
                minds = min(ds)
                if minds < 8 or (minds < 12 and self.isSafeCoordinate(food, gameState)):
                    temp.append((food, False))
        else:
            # Help invader, Eat the capsule
            if gameState.getAgentState(self.teamInvader).isPacman:
                for capsule in capsules:
                    ds = []
                    for i in range(self.maze_dim[1]):
                        if (self.boarder_mid[0], i) not in self.getWallList(gameState):
                            t = self.getMazeDistance(capsule, (self.boarder_mid[0], i))
                            ds.append(t)
                    minds = min(ds)
                    if minds < 8:
                        temp.append((capsule, True))
            # Don't Eat Capsule
            # Food: [d < 5 or (d < 8 and safe)]
            for food in foods:
                ds = []
                for i in range(self.maze_dim[1]):
                    if (self.boarder_mid[0], i) not in self.getWallList(gameState):
                        t = self.getMazeDistance(food, (self.boarder_mid[0], i))
                        ds.append(t)
                minds = min(ds)
                if minds < 5 or (minds < 8 and self.isSafeCoordinate(food, gameState)):
                    temp.append((food, False))

        nearestFoodDistance = 100
        nearestFood = None
        for (target, capsule) in temp:
            d = self.getMazeDistance(target, self.currentPosition)
            if capsule:
                d -= 5
            if d < nearestFoodDistance:
                nearestFoodDistance = d
                nearestFood = target

        if nearestFood:
            re = False
            for opponent in self.opponentIndices:
                # if in observable area
                if gameState.getAgentPosition(opponent) and (not gameState.getAgentState(opponent).isPacman) and gameState. \
                        getAgentState(opponent).scaredTimer <= 0:
                    avoid = gameState.getAgentPosition(opponent)
                    re = True
            if re:
                closestFoodProblem = AvoidProblem(gameState, self.currentPosition, opponent=avoid, goal=nearestFood, myzone=self.my_zone)
            else:
                closestFoodProblem = PositionSearchProblem(gameState, self.currentPosition, goal=nearestFood)
            actions = wastarSearch(closestFoodProblem, manhattanHeuristic)
            print("mode: stealFood")
            print(self.currentPosition)
            self.mode = "steal"
            if len(actions) > 0:
                return actions[0]
            else:
                return 'Stop'
        # No food to eat, go home and defend
        else:
            self.eatDefender = False
            return self.defend(gameState)

    def findInvader(self, gameState, foodEaten):
        re = False
        for opponent in self.opponentIndices:
            # if in observable area,
            if gameState.getAgentPosition(opponent) and (not gameState.getAgentState(opponent).isPacman) and \
                    gameState.getAgentState(opponent).scaredTimer <= 0:
                avoid = gameState.getAgentPosition(opponent)
                re = True
                break
        self.foodEaten = foodEaten
        if re:
            closestFoodProblem = AvoidProblem(gameState, self.currentPosition, opponent=avoid, goal=foodEaten, myzone=self.my_zone)
        else:
            closestFoodProblem = PositionSearchProblem(gameState, self.currentPosition, goal=foodEaten)

        if self.getMazeDistance(foodEaten, self.currentPosition) <= 12:
            actions = wastarSearch(closestFoodProblem, mazeDistanceHeuristic)
        else:
            actions = wastarSearch(closestFoodProblem, manhattanHeuristic)
        print("mode: findInvader")
        print(self.currentPosition)
        if len(actions) > 0:
            self.mode = "findInvader"
            return actions[0]
        else:
            self.searching = False
            return self.defend(gameState)

    def flee(self, gameState, enemyPosition):
        (x, y) = enemyPosition
        goal = list()
        goal_temp = [(x + 2, y), (x - 2, y), (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1), (x - 1, y - 1), (x, y + 2), (x, y - 2)]
        for g in goal_temp:
            if self.maze_dim[0] > g[0] >= 0 and self.maze_dim[1] > g[1] >= 0:
                if not self.walls[g[0]][g[1]]:
                    if self.getMazeDistance(g, enemyPosition) == 2:
                        goal.append(g)
        fleeProblem = FleeProblem(gameState, self.currentPosition, enemyPosition, goal=enemyPosition, goals=goal)
        actions = wastarSearch(fleeProblem, manhattanHeuristic_list)
        print("mode: flee")
        print(self.currentPosition)
        self.mode = "flee"
        if len(actions) > 0:
            return actions[0]
        else:
            return 'Stop'

    def hunt(self, gameState, invaders):
        """
        Actually it is defending food from opponent rather than hunting opponent
        hunting mode
        logic 1: Set the pacman's closet food as the goal
        logic 2: Set the pacman's position as the goal
        """

        ''' 1
        foodDefending, distance = self.closestObjectUsingPosition(self.getFoodYouAreDefending(gameState).asList(),
                                                                  self.opponentPosition)
        defendFoodProblem = PositionSearchProblem(gameState, gameState.getAgentPosition(self.index), goal=foodDefending)
        actions = wastarSearch(defendFoodProblem, manhattanHeuristic)
        if len(actions) > 0:
            return actions[0]
        else:
            return 'Stop'
        '''
        currentFood = self.getFoodYouAreDefending(gameState).asList()
        self.myFood = currentFood
        # Pick the nearest invader to hunt
        target = sorted(invaders,
                        key=lambda pos: pow(pos[0] - self.currentPosition[0], 2) + pow(pos[1] - self.currentPosition[1],
                                                                                       2))[0]
        # Can eat the invader after this action, the next action should be stealing food
        if abs(target[0] - self.currentPosition[0]) + abs(target[1] - self.currentPosition[1]) == 1:
            self.eatDefender = True
        re = False
        for opponent in self.opponentIndices:
            # if terrible ghost in observable area, defender should avoid it
            if gameState.getAgentPosition(opponent) and (not gameState.getAgentState(opponent).isPacman) and \
                    gameState.getAgentState(opponent).scaredTimer <= 0:
                avoid = gameState.getAgentPosition(opponent)
                re = True
                break
        self.target = target
        if re:
            defendFoodProblem = AvoidProblem(gameState, self.currentPosition, opponent=avoid, goal=target, myzone=self.my_zone)
        else:
            defendFoodProblem = PositionSearchProblem(gameState, self.currentPosition, goal=target)

        if self.getMazeDistance(target, self.currentPosition) <= 12:
            actions = wastarSearch(defendFoodProblem, mazeDistanceHeuristic)
        else:
            actions = wastarSearch(defendFoodProblem, manhattanHeuristic)
        print("mode: hunt")
        print(self.currentPosition)
        self.mode = "hunt"
        if len(actions) > 0:
            return actions[0]
        else:
            return Directions.STOP

    def farthestObjectUsingPosition(self, listOfObjects, currentPosition):
        farthestObj = None
        farthestDistance = 0
        for candidateObject in listOfObjects:
            if self.getMazeDistance(candidateObject, currentPosition) > farthestDistance:
                farthestDistance = self.getMazeDistance(candidateObject, currentPosition)
                farthestObj = candidateObject
        return farthestObj, farthestDistance

    def getWidthandHeight(self, gameState):
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        return width, height

    def getFeatures(self, gameState, action):
        pass

    def getWeights(self, gameState, action):
        pass


###################################
#        Helper Funcs
###################################
def wastarSearch(problem, heuristic):
    priorityFunc = lambda x: x[2] + heuristic(x[0], problem)

    # initialize a priority queue
    open = util.PriorityQueue()
    closed = []

    # Retrieve the init state
    init = (problem.getStartState(), ['Stop'], 0)
    open.push(init, priorityFunc(init))
    while not open.isEmpty():
        currNode = open.pop()
        currState = currNode[0]
        currPath = currNode[1]
        currPathCost = currNode[2]

        expandedThreshold = 360
        if problem._expanded == expandedThreshold:
            return currPath[1:]

        if problem.isGoalState(currState):
            return currPath[1:]
        else:
            closed.append(currState)
        successors = problem.getSuccessors(currState)

        if len(successors) > 0:
            for each in successors:
                # if each not in wallList:
                newState = each[0]
                newPathCost = currPathCost + each[2]
                if newState not in closed:
                    temp = (each[0], currPath + [each[1]], newPathCost)
                    open.update(temp, priorityFunc(temp))

    return []


def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def manhattanHeuristic_list(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    l = []
    for g in problem.goal:
        l.append(abs(xy1[0] - g[0]) + abs(xy1[1] - g[1]))
    return min(l)


def mazeDistanceHeuristic(position, problem, info = {}):
        prob = PositionSearchProblem(problem.gameState, start=position, goal=problem.goal,startState=None, warn=False, visualize=False)
        return len(breadthFirstSearch(prob))

###################################
#      Problem Gallery
###################################

# Search Problem Abstract Class
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


# PositionSearchProblem
class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, startState, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True, visualize=False):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = startState
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        self.gameState = gameState

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = game.Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = game.Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost


# Consider the opponent pacman and his surrounding area as walls
class FleeProblem(PositionSearchProblem):
    def __init__(self, gameState, startState, opponent, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True, visualize=False, goals = None):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        (x, y) = opponent
        self.walls[x][y] = True
        self.walls[x + 1][y] = True
        self.walls[x - 1][y] = True
        self.walls[x][y + 1] = True
        self.walls[x][y - 1] = True
        self.startState = startState
        if start != None: self.startState = start
        x = goal[0]
        y = goal[1]
        # self.goal = [(x + 2, y), (x - 2, y), (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1), (x - 1, y - 1)
        # , (x, y + 2) , (x, y - 2)]
        self.goal = goals
        self.costFn = costFn
        self.visualize = visualize

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state):
        isGoal = state in self.goal
        return isGoal


class AvoidProblem(FleeProblem):
    def __init__(self, gameState, startState, opponent, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True,
                 visualize=False, myzone=[]):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.gameState = gameState
        self.walls = gameState.getWalls()
        (x, y) = opponent
        self.walls[x][y] = True
        if x + 1 not in myzone:
            self.walls[x + 1][y] = True
        if x - 1 not in myzone:
            self.walls[x - 1][y] = True
        if x not in myzone:
            self.walls[x][y + 1] = True
            self.walls[x][y - 1] = True

        self.startState = startState
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state):
        isGoal = state == self.goal
        return isGoal


class TrackProblem(PositionSearchProblem):
    def __init__(self, gameState, startState, costFn=lambda x: 1, goal=(1, 1), opponentborder = 15, self_border = 16,height = 0,start=None, warn=True,
                 visualize=False, myZone = None, other = None):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        for i in range(height):
            self.walls[opponentborder][i] = True

        (goalx, goaly) = goal
        if goalx not in myZone:
            self.walls[goalx][goaly] = True
            self.walls[goalx][goaly+1] = True
            self.walls[goalx][goaly-1] = True

        if goalx+1 not in myZone:
            self.walls[goalx+1][goaly] = True
        if goalx-1 not in myZone:
            self.walls[goalx-1][goaly] = True

        self.startState = startState
        if start != None: self.startState = start
        self.goal = (self_border, goal[1])
        while self.walls[self.goal[0]][self.goal[1]]:
            self.goal = (self.goal[0], self.goal[1] + 1)
        self.costFn = costFn
        self.visualize = visualize

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

def generalGraphSearch(problem, structure):
    """
    Defines a general algorithm to search a graph.
    Parameters are structure, which can be any data structure with .push() and .pop() methods, and problem, which is the
    search problem.
    """

    # Push the root node/start into the data structure in this format: [(state, action taken, cost)]
    # The list pushed into the structure for the second node will look something like this:
    # [(root_state, "Stop", 0), (new_state, "North", 1)]
    structure.push([(problem.getStartState(), "Stop", 0)])

    # Initialise the list of visited nodes to an empty list
    visited = []

    # While the structure is not empty, i.e. there are still elements to be searched,
    while not structure.isEmpty():
        # get the path returned by the data structure's .pop() method
        path = structure.pop()

        # The current state is the first element in the last tuple of the path
        # i.e. [(root_state, "Stop", 0), (new_state, "North", 1)][-1][0] = (new_state, "North", 1)[0] = new_state
        curr_state = path[-1][0]

        # if the current state is the goal state,
        if problem.isGoalState(curr_state):
            # return the actions to the goal state
            # which is the second element for each tuple in the path, ignoring the first "Stop"
            return [x[1] for x in path][1:]

        # if the current state has not been visited,
        if curr_state not in visited:
            # mark the current state as visited by appending to the visited list
            visited.append(curr_state)

            # for all the successors of the current state,
            for successor in problem.getSuccessors(curr_state):
                # successor[0] = (state, action, cost)[0] = state
                # if the successor's state is unvisited,
                if successor[0] not in visited:
                    # Copy the parent's path
                    successorPath = path[:]
                    # Set the path of the successor node to the parent's path + the successor node
                    successorPath.append(successor)
                    # Push the successor's path into the structure
                    structure.push(successorPath)

    # If search fails, return False
    return False

def breadthFirstSearch(problem):
    # Initialize an empty Queue
    queue = util.Queue()

    # BFS is general graph search with a Queue as the data structure
    return generalGraphSearch(problem, queue)
