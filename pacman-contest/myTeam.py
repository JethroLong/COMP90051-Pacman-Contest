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
import sys

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################
from graphicsDisplay import InfoPane


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

class DummyAgent(CaptureAgent):
    
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """
    
    mode = ""
    # invader home mode, invader hunting mode, invader power mode, invader rereat mode
    
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
    
    def getHome(self, gameState):
        width = gameState.data.layout.width
        if self.red:
            return int(width / 2), "<"
        else:
            return int(width / 2), ">="
    
    def getWallList(self, gameState):
        return gameState.getWalls().asList()
    
    def getOpponentList(self, gameState):
        opponentList = []
        for opponentIndex in self.getOpponents(gameState):
            if not gameState.getAgentState(opponentIndex).isPacman:
                opponentList.append(gameState.getAgentPosition(opponentIndex))
        notNoneOpponentList = []
        for i in range(len(opponentList)):
            if opponentList[i] is not None:
                notNoneOpponentList.append(opponentList[i])
        return notNoneOpponentList
    
    def getOpponentPacmanList(self, gameState):
        opponentPacmanList = []
        for opponentIndex in self.getOpponents(gameState):
            if gameState.getAgentState(opponentIndex).isPacman:
                opponentPacmanList.append(gameState.getAgentPosition(opponentIndex))
        notNoneOpponentPacmanList = []
        for i in range(len(opponentPacmanList)):
            if opponentPacmanList[i] is not None:
                notNoneOpponentPacmanList.append(opponentPacmanList[i])
        return notNoneOpponentPacmanList
    
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
            if not gameState.getAgentState(opponentIndex).isPacman:
                opponentList.append(gameState.getAgentPosition(opponentIndex))
        notNoneOpponentList = []
        for i in range(len(opponentList)):
            if opponentList[i] is not None:
                notNoneOpponentList.append(opponentList[i])
        if (x, y + 1) not in wallList and (x, y + 1) not in notNoneOpponentList:
            legalAction.append("North")
        if (x, y - 1) not in wallList and (x, y - 1) not in notNoneOpponentList:
            legalAction.append("South")
        if (x - 1, y) not in wallList and (x - 1, y) not in notNoneOpponentList:
            legalAction.append("West")
        if (x + 1, y) not in wallList and (x + 1, y) not in notNoneOpponentList:
            legalAction.append("East")
        if (x, y + 1) not in notNoneOpponentList and (x, y - 1) not in notNoneOpponentList and (x + 1, y) not in notNoneOpponentList and (x - 1, y) not in notNoneOpponentList:
            legalAction.append("Stop")
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

    def breadthFirstSearch(self, problem, avoidCoordinate, isRed, boardWidth, wallList, opponentList):
        
        for element in opponentList:
            if element not in wallList:
                wallList.append(element)
        
        open = util.Queue()
        init = (problem.getStartState(), ['Stop'], 0)
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
        for element in opponentList:
            if element not in wallList:
                wallList.append(element)
        open = util.Stack()
        initState = (problem.getStartState(), ['Stop'], 0)
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




######################################
#            WA* Agents
######################################


class WaStarInvader(DummyAgent):
    
    """
    invader home mode
    invader hunting mode
    invader power mode
    invader retreat mode
    """
    
    def isSafeCoordinate(self, coordinate, gameState):
        """
        Decide whether the given coordinate is safe or not,
        i.e. whether there exists at least two ways back home.
        """
        x, y = coordinate[0], coordinate[1]
        legalAction = self.getLegalActionOfPosition(coordinate, gameState)
        if len(legalAction) <= 1:
            return False
        if len(legalAction) == 2 and "Stop" in legalAction:
            return False
        nonStopLegalAction = []
        for action in legalAction:
            if action != "Stop":
                nonStopLegalAction.append(action)
        newStartingPoint = []
        for action in nonStopLegalAction:
            if action == "East":
                newStartingPoint.append((x + 1, y))
            elif action == "West":
                newStartingPoint.append((x - 1, y))
            elif action == "North":
                newStartingPoint.append((x, y + 1))
            elif action == "South":
                newStartingPoint.append((x, y - 1))
        number_of_escape_path = 0
        for startingPoint in newStartingPoint:
            bfsProblem = PositionSearchProblem(gameState, startingPoint)
            boardWidth, boardHeight = self.getWidthandHeight(gameState)
            path = self.depthFirstSearch(bfsProblem, coordinate, self.red, boardWidth, self.getWallList(gameState), self.getOpponentList(gameState))
            if len(path) != 0:
                number_of_escape_path += 1
            if number_of_escape_path > 1:
                return True
        return False
    
    def chooseAction(self, gameState):
    
        scaredTime = 0
        opponentDict = dict()
        for opponentIndex in self.getOpponents(gameState):
            if not gameState.getAgentState(opponentIndex).isPacman:
                tempScaredTime = gameState.data.agentStates[opponentIndex].scaredTimer
                opponentDict[gameState.getAgentPosition(opponentIndex)] = tempScaredTime
                if tempScaredTime > scaredTime:
                    scaredTime = tempScaredTime

        opponentList = self.getOpponentList(gameState)
        currentPosition = gameState.getAgentPosition(self.index)
        closestOpponent, closestOpponentDistance = self.closestObjectUsingPosition(opponentList, currentPosition)
        
        self.updateMode(gameState, scaredTime, closestOpponentDistance)
        
        print("============INVADER============")
        print("Mode: " + self.mode)
        print("Position: " + str(gameState.getAgentPosition(self.index)))
        
        foodList = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)
        opponentPacmanList = self.getOpponentPacmanList(gameState)
        print("Capsule List: " + str(capsuleList))
        print("Opponent List: " + str(opponentList))
        
        wallList = self.getWallList(gameState)
        original_wall_grids = gameState.data.layout.walls
        updatedGameState = gameState.deepCopy()
        grid_width, grid_height = self.getWidthandHeight(gameState)
        wall_grids = gameState.getWalls().data
        currentPosition = gameState.getAgentPosition(self.index)
        
        # DETECT OPPONENT AROUND ME
        if len(opponentList) != 0 and self.mode != "invader power mode":
            print("Number of Opponent Around Me: " + str(len(opponentList)))
            for candidateOpponent in opponentList:
                distance = self.getMazeDistance(candidateOpponent, currentPosition)
                print("Opponent " + str(candidateOpponent) + " Distance: " + str(distance))
                if candidateOpponent not in wallList:
                    wallList.append(candidateOpponent)
                    wall_grids[candidateOpponent[0]][candidateOpponent[1]] = True
                if distance <= 2:
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
        
        myScaredTime = 0
        if not updatedGameState.getAgentState(self.index).isPacman:
            myScaredTime = updatedGameState.data.agentStates[self.index].scaredTimer
        if len(opponentPacmanList) != 0 and self.mode == "invader home mode" and myScaredTime > 0:
            for candidateOpponentPacman in opponentPacmanList:
                distance = self.getMazeDistance(candidateOpponentPacman, currentPosition)
                print("Opponent Pacman " + str(candidateOpponentPacman) + "Distance: " + str(distance))
                if candidateOpponentPacman not in wallList:
                    wallList.append(candidateOpponentPacman)
                    wall_grids[candidateOpponentPacman[0]][candidateOpponentPacman[1]] = True
                if distance <= 2:
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

        combined_wall_grid = game.Grid(grid_width, grid_height, False)
        for i in range(grid_width):
            for j in range(grid_height):
                combined_wall_grid.data[i][j] = wall_grids[i][j]
        updatedGameState.data.layout.walls = combined_wall_grid
            
        if self.mode == "invader home mode":
            for element in capsuleList:
                if element not in foodList:
                    foodList.append(element)
            huntingPacmanScoreThreshold = 3
            score = self.getScore(updatedGameState)
            if score > huntingPacmanScoreThreshold:
                for element in opponentPacmanList:
                    if element not in foodList:
                        foodList.append(element)
            closestFood, distance = self.closestObject(foodList, updatedGameState)
            closestFoodProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestFood)
            actions = wastarSearch(closestFoodProblem, manhattanHeuristic)
            self.updateScore(updatedGameState)
            print("Goal: " + str(closestFood))
            if closestFood in foodList:
                print("Goal Type: Food")
            elif closestFood in capsuleList:
                print("Goal Type: Capsule")
            if len(actions) == 0:
                actions.append("Stop")
            print("Action: " + actions[0])
            print("===============================")
            print()
            print()
            updatedGameState.data.layout.walls = original_wall_grids
            gameState = updatedGameState
            return actions[0]
        
        elif self.mode == "invader hunting mode" and len(opponentList) == 0:
            foodDistanceThreshold = 8
            goodList = []
            for element in foodList:
                if element not in goodList and self.getMazeDistance(currentPosition, element) < foodDistanceThreshold:
                    goodList.append(element)
            for element in capsuleList:
                if element not in goodList and self.getMazeDistance(currentPosition, element) < foodDistanceThreshold:
                    goodList.append(element)
            huntingPacmanScoreThreshold = 3
            score = self.getScore(updatedGameState)
            if score > huntingPacmanScoreThreshold:
                for element in opponentPacmanList:
                    if element not in goodList and self.getMazeDistance(currentPosition, element) < foodDistanceThreshold:
                        goodList.append(element)
            if len(goodList) == 0:
                print("Trigger invader retreat mode")
                width, height = self.getWidthandHeight(updatedGameState)
                homeWidth = int(width / 2)
                candidateHomeList = []
                if self.red:
                    for i in range(1, height):
                        if (homeWidth - 1, i) not in wallList:
                            candidateHomeList.append((homeWidth - 1, i))
                else:
                    for i in range(1, height):
                        if (homeWidth + 2, i) not in wallList:
                            candidateHomeList.append((homeWidth + 2, i))
                closestHome, distance = self.closestObject(candidateHomeList, updatedGameState)
                goHomeProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestHome)
                actions = wastarSearch(goHomeProblem, manhattanHeuristic)
                self.updateScore(updatedGameState)
                print("Goal: " + str(closestHome))
                print("Goal Type: Closest Home")
                if len(actions) == 0:
                    actions.append("Stop")
                print("Action: " + actions[0])
                print("===============================")
                print()
                print()
                updatedGameState.data.layout.walls = original_wall_grids
                gameState = updatedGameState
                return actions[0]
            closestFood, distance = self.closestObject(goodList, updatedGameState)
            closestFoodProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestFood)
            actions = wastarSearch(closestFoodProblem, manhattanHeuristic)
            self.updateScore(updatedGameState)
            print("Goal: " + str(closestFood))
            if closestFood in foodList:
                print("Goal Type: Food")
            elif closestFood in capsuleList:
                print("Goal Type: Capsule")
            if len(actions) == 0:
                actions.append("Stop")
            print("Action: " + actions[0])
            print("===============================")
            print()
            print()
            updatedGameState.data.layout.walls = original_wall_grids
            gameState = updatedGameState
            return actions[0]
        
        elif self.mode == "invader hunting mode" and len(opponentList) != 0:
            safeList = []
            for foodCoordinate in foodList:
                if self.isSafeCoordinate(foodCoordinate, updatedGameState):
                    safeList.append(foodCoordinate)
            for capsuleCoordinate in capsuleList:
                safeList.append(capsuleCoordinate)
            if len(safeList) == 0:
                print("Trigger invader retreat mode")
                width, height = self.getWidthandHeight(updatedGameState)
                homeWidth = int(width / 2)
                candidateHomeList = []
                if self.red:
                    for i in range(1, height):
                        if (homeWidth - 1, i) not in wallList:
                            candidateHomeList.append((homeWidth - 1, i))
                else:
                    for i in range(1, height):
                        if (homeWidth + 2, i) not in wallList:
                            candidateHomeList.append((homeWidth + 2, i))
                closestHome, distance = self.closestObject(candidateHomeList, updatedGameState)
                goHomeProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestHome)
                actions = wastarSearch(goHomeProblem, manhattanHeuristic)
                self.updateScore(updatedGameState)
                print("Goal: " + str(closestHome))
                print("Goal Type: Closest Home")
                if len(actions) == 0:
                    actions.append("Stop")
                print("Action: " + actions[0])
                print("===============================")
                print()
                print()
                updatedGameState.data.layout.walls = original_wall_grids
                gameState = updatedGameState
                return actions[0]
            closestSafe, distance = self.closestObject(safeList, updatedGameState)
            if closestSafe is not None:
                closestFoodProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestSafe)
                actions = wastarSearch(closestFoodProblem, manhattanHeuristic)
            else:
                width, height = self.getWidthandHeight(updatedGameState)
                homeWidth = int(width / 2)
                candidateHomeList = []
                if self.red:
                    for i in range(1, height):
                        if (homeWidth - 1, i) not in wallList:
                            candidateHomeList.append((homeWidth - 1, i))
                else:
                    for i in range(1, height):
                        if (homeWidth + 2, i) not in wallList:
                            candidateHomeList.append((homeWidth + 2, i))
                closestHome, distance = self.closestObject(candidateHomeList, updatedGameState)
                goHomeProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestHome)
                actions = wastarSearch(goHomeProblem, manhattanHeuristic)
            self.updateScore(updatedGameState)
            print("Goal: " + str(closestSafe))
            if closestSafe in foodList:
                print("Goal Type: Safe Food")
            elif closestSafe in capsuleList:
                print("Goal Type: Safe Capsule")
            else:
                print("Goal Type: Closest Home")
            if len(actions) == 0:
                actions.append("Stop")
            print("Action: " + actions[0])
            print("===============================")
            print()
            print()
            updatedGameState.data.layout.walls = original_wall_grids
            gameState = updatedGameState
            return actions[0]
        
        elif self.mode == "invader power mode":
            for element in capsuleList:
                if element not in foodList:
                    foodList.append(element)
            scaredTimerCountingDownThreshold = 5
            for element in opponentList:
                if element not in foodList and opponentDict[element] >= scaredTimerCountingDownThreshold:
                    foodList.append(element)
            # huntingPacmanScoreThreshold = 5
            # score = self.getScore(updatedGameState)
            # if score > huntingPacmanScoreThreshold:
            #     for element in opponentPacmanList:
            #         if element not in foodList:
            #             foodList.append(element)
            if len(foodList) == 0:
                print("Trigger invader retreat mode")
                width, height = self.getWidthandHeight(updatedGameState)
                homeWidth = int(width / 2)
                candidateHomeList = []
                if self.red:
                    for i in range(1, height):
                        if (homeWidth - 1, i) not in wallList:
                            candidateHomeList.append((homeWidth - 1, i))
                else:
                    for i in range(1, height):
                        if (homeWidth + 2, i) not in wallList:
                            candidateHomeList.append((homeWidth + 2, i))
                closestHome, distance = self.closestObject(candidateHomeList, updatedGameState)
                goHomeProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestHome)
                actions = wastarSearch(goHomeProblem, manhattanHeuristic)
                self.updateScore(updatedGameState)
                print("Goal: " + str(closestHome))
                print("Goal Type: Closest Home")
                if len(actions) == 0:
                    actions.append("Stop")
                print("Action: " + actions[0])
                print("===============================")
                print()
                print()
                updatedGameState.data.layout.walls = original_wall_grids
                gameState = updatedGameState
                return actions[0]
            closestFood, distance = self.closestObject(foodList, updatedGameState)
            closestFoodProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestFood)
            actions = wastarSearch(closestFoodProblem, manhattanHeuristic)
            self.updateScore(updatedGameState)
            print("Goal: " + str(closestFood))
            if closestFood in capsuleList:
                print("Goal Type: Capsule")
            elif closestFood in opponentList:
                print("Goal Type: Opponent Ghost")
            else:
                print("Goal Type: Food")
            if len(actions) == 0:
                actions.append("Stop")
            print("Action: " + actions[0])
            print("Capsule Eaten: " + str(updatedGameState.data._capsuleEaten))
            scaredTime = 0
            for opponentIndex in self.getOpponents(updatedGameState):
                if not updatedGameState.getAgentState(opponentIndex).isPacman:
                    scaredTime = updatedGameState.data.agentStates[opponentIndex].scaredTimer
                    break
            print("Scared Timer: " + str(scaredTime))
            print("===============================")
            print()
            print()
            updatedGameState.data.layout.walls = original_wall_grids
            gameState = updatedGameState
            return actions[0]
        
        elif self.mode == "invader retreat mode":
            width, height = self.getWidthandHeight(updatedGameState)
            homeWidth = int(width / 2)
            candidateHomeList = []
            if self.red:
                for i in range(1, height):
                    if (homeWidth - 1, i) not in wallList:
                        candidateHomeList.append((homeWidth - 1, i))
            else:
                for i in range(1, height):
                    if (homeWidth + 2, i) not in wallList:
                        candidateHomeList.append((homeWidth + 2, i))
            closestHome, distance = self.closestObject(candidateHomeList, updatedGameState)
            goHomeProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestHome)
            actions = wastarSearch(goHomeProblem, manhattanHeuristic)
            self.updateScore(updatedGameState)
            print("Goal: " + str(closestHome))
            print("Goal Type: Closest Home")
            if len(actions) == 0:
                actions.append("Stop")
            print("Action: " + actions[0])
            print("===============================")
            print()
            print()
            updatedGameState.data.layout.walls = original_wall_grids
            gameState = updatedGameState
            return actions[0]
           
    def updateScore(self, gameState):
        gameState.data.score = len(gameState.getRedFood().asList()) - len(gameState.getBlueFood().asList())
    
    """
    self.updateMode(gameState)
        print(gameState.data._capsuleEaten)
        for opponentIndex in self.getOpponents(gameState):
            if not gameState.getAgentState(opponentIndex).isPacman:
                print(gameState.data.agentStates[opponentIndex].scaredTimer)
    """
    
    def updateMode(self, gameState, scaredTime, closestOpponentDistance):
        boardWidth, boardHeight = self.getWidthandHeight(gameState)
        currentPosition = gameState.getAgentPosition(self.index)

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
            scaredTimerCountingDownThreshold = 5
            if gameState.data._capsuleEaten is None and scaredTime >= scaredTimerCountingDownThreshold:
                self.mode = "invader power mode"
                return
            if gameState.data._capsuleEaten is None and scaredTime < scaredTimerCountingDownThreshold:
                self.mode = "invader hunting mode"
            escapeDistanceThreshold = 3
            escapeScoreThreshold = 0
            if closestOpponentDistance <= escapeDistanceThreshold:
                currentScore = self.getScore(gameState)
                if (currentScore > escapeScoreThreshold and self.red) or (currentScore < -1 * escapeScoreThreshold and not self.red):
                    self.mode = "invader retreat mode"
                else:
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
    mode = "goDefendingFood"
    initialPosition = None
    oppInitialPosition = None
    myZone = None  # Safe space
    opponentPosition = None

    def chooseAction(self, gameState):
        # The first time to choose an action
        if not self.initialPosition:
            self.initialPosition = gameState.getAgentPosition(self.index)
            width, height = self.getWidthandHeight(gameState)
            if self.initialPosition == (1, 1):
                self.oppInitialPosition = (width - 2, height - 2)
                self.myZone = range(int(width / 2))
            else:
                self.oppInitialPosition = (1, 1)
                self.myZone = range(int(width / 2), int(width))
        opponents = self.getOpponents(gameState)  # the indexes of opponents
    
        # To determine the mode
        for opponent in opponents:
            # if in observable area
            if gameState.getAgentPosition(opponent):
                # if the agent is not scared
                if gameState.getAgentState(self.index).scaredTimer <= 0:
                    # if the opponent invades
                    if gameState.getAgentPosition(opponent)[0] in self.myZone:
                        self.mode = "huntOpponent"
                        self.opponentPosition = gameState.getAgentPosition(opponent)
                        break
                    else:
                        self.mode = "goDefendingFood"
                else:
                    self.opponentPosition = gameState.getAgentPosition(opponent)
                    self.mode = "flee"
                    break
            # if no opponents in observable area
            else:
                self.mode = "goDefendingFood"
    
        # When the agent does not find any invader
        if self.mode == "goDefendingFood":
            currentPosition = gameState.getAgentPosition(self.index)
            foodDefending, distance = self.closestObjectUsingPosition(self.getFoodYouAreDefending(gameState).asList(),
                                                                      self.oppInitialPosition)
            defendFoodProblem = PositionSearchProblem(gameState, gameState.getAgentPosition(self.index), goal=foodDefending)
            actions = wastarSearch(defendFoodProblem, manhattanHeuristic)
            if len(actions) > 0:
                return actions[0]
            else:
                return 'Stop'
        '''
        Actually it is defending food from opponent rather than hunting opponent
        hunting mode
        logic 1: Set the pacman's closet food as the goal
        logic 2: Set the pacman's position as the goal
        '''
        if self.mode == "huntOpponent":
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
            defendFoodProblem = PositionSearchProblem(gameState, gameState.getAgentPosition(self.index),
                                                      goal=self.opponentPosition)
            actions = wastarSearch(defendFoodProblem, manhattanHeuristic)
            if len(actions) > 0:
                return actions[0]
            else:
                return 'Stop'
    
        if self.mode == "flee":
            foodDefending, distance = self.closestObjectUsingPosition(self.getFoodYouAreDefending(gameState).asList(),
                                                                      self.opponentPosition)
            fleeProblem = FleeProblem(gameState, gameState.getAgentPosition(self.index), self.opponentPosition,
                                      goal=foodDefending)
            actions = wastarSearch(fleeProblem, manhattanHeuristic)
            if len(actions) > 0:
                return actions[0]
            else:
                return 'Stop'

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
        
        if problem.isGoalState(currState):
            return currPath[1:]
        else:
            closed.append(currState)
        successors = problem.getSuccessors(currState)
        
        if len(successors) > 0:
            for each in successors:
                #if each not in wallList:
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

def heuristic_switcher(state, gameState):
    pass

def nullHeuristic():
    return 0

def foodHeuristic():
    pass

def capsuleHeuristic():
    pass

def manhattanDistance(point1, point2):
    return abs(point1[0]-point2[0]) + abs(point1[1]-point2[1])

def findNearestFood(state, gameState):
    pass

def findFurthestFood(state, gameState):
    pass

def findNearestGhost(state, gameState):
    pass


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
    def __init__(self, gameState, startState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
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
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal
        
        if str(self.goal).startswith("HOME"):
            param1, param2 = str(self.goal).split(" ")[1], str(self.goal).split(" ")[2]
            if param2 == "<":
                if state[0] < param1:
                    return True
                else:
                    return False
            elif param2 == ">=":
                if state[0] >= param1:
                    return True
                else:
                    return False

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

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
            x,y = state
            dx, dy = game.Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
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
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = game.Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

# Consider the opponent pacman and his surrounding area as walls
class FleeProblem(PositionSearchProblem):
    def __init__(self, gameState, startState, opponent, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        (x,y) = opponent
        self.walls[x][y] = True
        self.walls[x+1][y] = True
        self.walls[x-1][y] = True
        self.walls[x][y+1] = True
        self.walls[x][y-1] = True
        self.startState = startState
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE