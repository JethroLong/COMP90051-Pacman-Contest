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
from util import nearestPoint
import re, os


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
        
    def getFoodXCoordinateClosestToBorder(self, gameState):
        foodList = self.getFood(gameState).asList()
        if self.red:
            x = sys.maxsize
        else:
            x = -1 * sys.maxsize + 1
        for foodCoordinate in foodList:
            if self.red:
                if foodCoordinate[0] < x:
                    x = foodCoordinate[0]
            else:
                if foodCoordinate[0] > x:
                    x = foodCoordinate[0]
        return x
    
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
            legalAction.append("North")
        if (x, y - 1) not in wallList and (x, y - 1) not in opponentList:
            legalAction.append("South")
        if (x - 1, y) not in wallList and (x - 1, y) not in opponentList:
            legalAction.append("West")
        if (x + 1, y) not in wallList and (x + 1, y) not in opponentList:
            legalAction.append("East")
        if (x, y + 1) not in opponentList and (x, y - 1) not in opponentList and (x + 1, y) not in opponentList and (x - 1, y) not in opponentList:
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

    def depthFirstSearchCycleDetecter(self, problem):
        
        open = util.Stack()
        initState = (problem.getStartState(), ['Stop'], 0)
        open.push(initState)
        closed = []
        initial = True
        while not open.isEmpty():
            currState = open.pop()
            currPos = currState[0]
            currPath = currState[1]
            currCost = currState[2]
        
            if currPos == problem.startState and not initial:
                return True
            else:
                if initial:
                    initial = False
                closed.append(currPos)
            if currState not in closed:
                successors = problem.getSuccessors(currPos)
                if len(successors) > 0:
                    for each in successors:
                        if each[0] not in closed:
                            temp = (each[0], currPath + [each[1]], currCost + each[2])
                            open.push(temp)
        return False




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
    
    """
    def isRoadBlockingCapsule(self, coordinate, gameState):
    
        # Decide whether the given capsule blocks the road to reach certain food,
        # i.e. whether this capsule forms a cycle with walls.
        
        tempGameState = gameState.deepCopy()
        grid_width, grid_height = self.getWidthandHeight(gameState)
        wall_grids = gameState.getWalls().data
        capsuleList = self.getCapsules(gameState)
        notNoneCapsuleList = []
        for capsule in capsuleList:
            if capsule is not None:
                notNoneCapsuleList.append(capsule)
        for element in notNoneCapsuleList:
            if element != coordinate:
                wall_grids[element[0]][element[1]] = True
        temp_combined_wall_grid = game.Grid(grid_width, grid_height, False)
        for i in range(grid_width):
            for j in range(grid_height):
                temp_combined_wall_grid.data[i][j] = not wall_grids[i][j]
        tempGameState.data.layout.walls = temp_combined_wall_grid
        
        dfsProblem = PositionSearchProblem(tempGameState, coordinate)
        boardWidth, boardHeight = self.getWidthandHeight(tempGameState)
        return self.depthFirstSearchCycleDetecter(dfsProblem)
    """
    
    def chooseLegalRandomAction(self, currentPosition, wallList):
        actions = []
        x, y = currentPosition[0], currentPosition[1]
        if (x + 1, y) not in wallList:
            actions.append("East")
        if (x - 1, y) not in wallList:
            actions.append("West")
        if (x, y + 1) not in wallList:
            actions.append("North")
        if (x, y - 1) not in wallList:
            actions.append("South")
        if len(actions) == 0:
            actions.append("Stop")
        if actions[0] is None or actions[0] == "None":
            print("ERROR !!!")
        return actions
    
    def bestAvoidGhostAction(self, currentPosition, wallList, opponentList):
        closestOpponent = None
        closestOpponentDistance = sys.maxsize
        for opponent in opponentList:
            tempDistance = self.getMazeDistance(currentPosition, opponent)
            if tempDistance < closestOpponentDistance:
                closestOpponent = opponent
                closestOpponentDistance = tempDistance
        if closestOpponent is not None:
            distanceToGhost = -1 * sys.maxsize + 1
            wisestAction = None
            x, y = currentPosition[0], currentPosition[1]
            if (x + 1, y) not in wallList:
                tempDistance = self.getMazeDistance((x + 1, y), closestOpponent)
                if tempDistance > distanceToGhost:
                    distanceToGhost = tempDistance
                    wisestAction = "East"
            if (x - 1, y) not in wallList:
                tempDistance = self.getMazeDistance((x - 1, y), closestOpponent)
                if tempDistance > distanceToGhost:
                    distanceToGhost = tempDistance
                    wisestAction = "West"
            if (x, y + 1) not in wallList:
                tempDistance = self.getMazeDistance((x, y + 1), closestOpponent)
                if tempDistance > distanceToGhost:
                    distanceToGhost = tempDistance
                    wisestAction = "North"
            if (x, y - 1) not in wallList:
                tempDistance = self.getMazeDistance((x, y - 1), closestOpponent)
                if tempDistance > distanceToGhost:
                    distanceToGhost = tempDistance
                    wisestAction = "South"
            if wisestAction is None:
                return [self.chooseLegalRandomAction(currentPosition, wallList)[0]]
            return [wisestAction]
        else:
            return [self.chooseLegalRandomAction(currentPosition, wallList)[0]]
    
    def isSafeCoordinate(self, coordinate, gameState):
        """
        Decide whether the given coordinate is safe or not,
        i.e. whether there exists at least two ways back home.
        """
        wallList = gameState.getWalls().asList()
        if coordinate in wallList:
            return False
        """
        opponentList = self.getOpponentList(gameState)
        closestOpponentDistance = sys.maxsize
        for opponent in opponentList:
            if opponent is not None:
                tempDistance = self.getMazeDistance(opponent, coordinate)
                if tempDistance < closestOpponentDistance:
                    closestOpponentDistance = tempDistance
        if self.isWallCornerCoordinate(coordinate, wallList) and closestOpponentDistance >= 3:
            return True
        """
        x, y = coordinate[0], coordinate[1]
        legalAction = self.getLegalActionOfPosition(coordinate, gameState)
        if len(legalAction) <= 1:
            return False
        if len(legalAction) == 2 and "Stop" in legalAction:
            return False
        capsuleList = self.getCapsuleList(gameState)
        for capsule in capsuleList:
            if self.getMazeDistance(coordinate, capsule) <= 2: # Only applicable for maze distance
                return True
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
    
    def isWallCornerCoordinate(self, coordinate, wallList):
        pass
        
    
    def retreat(self, gameState):
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
        self.updateScore(gameState)
        print("Goal: " + str(closestHome))
        print("Goal Type: Closest Home")
        if len(actions) == 0:
            # actions.append("Stop")
            # Don't Stop, stop is the most stupidest move
            print("Empty Action List, Random Select Legal Actions")
            # actions = self.chooseLegalRandomAction(currentPosition, wallList)
            actions = self.bestAvoidGhostAction(currentPosition, wallList, self.getOpponentList(gameState))
        print("Action: " + actions[0])
        print("===============================")
        print()
        print()
        if actions[0] is None or actions[0] == "None":
            print("ERROR !!!")
        return actions
    
    def chooseAction(self, gameState):
    
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
        
        print("============INVADER============")
        print("Mode: " + self.mode)
        print("Position: " + str(gameState.getAgentPosition(self.index)))
        
        foodList = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)
        opponentPacmanList = self.getOpponentPacmanList(gameState)
        print("Capsule List: " + str(capsuleList))
        print("Opponent List: " + str(opponentList))
        print("Opponent Pacman List: " + str(opponentPacmanList))
        if closestOpponent is not None:
            print("Closest Opponent Ghost: " + str(closestOpponent) + ": " + str(closestOpponentDistance))
        
        wallList = self.getWallList(gameState)
        original_wall_grids = gameState.data.layout.walls
        updatedGameState = gameState.deepCopy()
        grid_width, grid_height = self.getWidthandHeight(gameState)
        wall_grids = gameState.getWalls().data
        
        # DETECT OPPONENT AROUND ME
        if len(opponentList) != 0 and numberOfScaredGhost != len(list(opponentDict.keys())):
            # There are ghosts around me, and at least one of them is not scared.
            print("Number of Opponent Around Me: " + str(len(opponentList)))
            print("Number of Scared Opponent Around Me: " + str(numberOfScaredGhost))
            for candidateOpponent in opponentList:
                for key in opponentDict:
                    if candidateOpponent == key[1] and opponentDict[key] <= countAsScaredThreshold:
                        distance = self.getMazeDistance(candidateOpponent, currentPosition)
                        print("Not Scared Opponent " + str(candidateOpponent) + " Distance: " + str(distance))
                        if candidateOpponent not in wallList:
                            wallList.append(candidateOpponent)
                            wall_grids[candidateOpponent[0]][candidateOpponent[1]] = True
                        considerGhostSurroundingAreaDistanceThreshold = 2
                        if distance <= considerGhostSurroundingAreaDistanceThreshold:
                            # Right now, consider 4 surrounding cells as wall
                            # Can also consider all 8 surrounding cells as wall # IDEA
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
        
        myScaredTime = 0 # used if I am a ghost at home territory.
        if not updatedGameState.getAgentState(self.index).isPacman:
            myScaredTime = updatedGameState.data.agentStates[self.index].scaredTimer
        
        if len(opponentPacmanList) != 0 and self.mode == "invader home mode" and myScaredTime > 0:
            print("My Scared Timer: " + str(myScaredTime))
            for candidateOpponentPacman in opponentPacmanList:
                distance = self.getMazeDistance(candidateOpponentPacman, currentPosition)
                print("Opponent Pacman " + str(candidateOpponentPacman) + "Distance: " + str(distance))
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

        if gameState.getAgentState(self.index).isPacman:
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
        
        
        if self.mode == "invader home mode":
            for element in capsuleList:
                if element not in foodList: # and self.isRoadBlockingCapsule(element, gameState):
                    foodList.append(element)
            huntingPacmanScoreThreshold = 3
            score = self.getScore(updatedGameState)
            if score >= huntingPacmanScoreThreshold and myScaredTime == 0:
                print("Optimistic Score & Nearby Pacman Present, Hunt It! ")
                print("Score: " + str(score))
                print("Nearby Pacman List: " + str(opponentPacmanList))
                for element in opponentPacmanList:
                    if element not in foodList:
                        foodList.append(element)
                        
            if len(foodList) == 0:
                actions = self.chooseLegalRandomAction(currentPosition, wallList)
                print("Empty food list, randomly choose legal action")
                return actions[0]
            
            closestFood, distance = self.closestObject(foodList, updatedGameState)
            closestFoodProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestFood)
            actions = wastarSearch(closestFoodProblem, manhattanHeuristic)
            self.updateScore(updatedGameState)
            print("Goal: " + str(closestFood))
            if closestFood in foodList:
                print("Goal Type: Food")
            elif closestFood in capsuleList:
                print("Goal Type: Capsule")
            else:
                print("Goal Type: Nearby Pacman in Home Territory")
            if len(actions) == 0:
                print("Food list not empty, however, no action available, randomly choose legal action")
                actions = self.chooseLegalRandomAction(currentPosition, wallList)
            print("Action: " + actions[0])
            print("===============================")
            print()
            print()
            updatedGameState.data.layout.walls = original_wall_grids
            gameState = updatedGameState
            if actions[0] is None or actions[0] == "None":
                print("ERROR !!!")
            return actions[0]
        
        
        elif self.mode == "invader hunting mode" and len(opponentList) == 0:
            # If I currently carries no food, than do my best
            # Otherwise, find food within certain distance threshold.
            if updatedGameState.getAgentState(self.index).numCarrying == 0:
                foodDistanceThreshold = sys.maxsize
            else:
                foodDistanceThreshold = 8
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
                if element not in goodList and tempDistance < foodDistanceThreshold: # and self.isRoadBlockingCapsule(element, updatedGameState):
                    goodList.append(element)
                    if tempDistance < closestDistance:
                        closestDistance = tempDistance
                        closestFood = element
            huntingPacmanScoreThreshold = 3
            score = self.getScore(updatedGameState)
            if score >= huntingPacmanScoreThreshold:
                for element in opponentPacmanList:
                    tempDistance = self.getMazeDistance(currentPosition, element)
                    if element not in goodList and tempDistance < foodDistanceThreshold:
                        goodList.append(element)
                        if tempDistance < closestDistance:
                            closestDistance = tempDistance
                            closestFood = element
            if len(goodList) == 0 or closestFood is None:
                print("Mode Changes: invader retreat mode")
                actions = self.retreat(updatedGameState)
                updatedGameState.data.layout.walls = original_wall_grids
                gameState = updatedGameState
                if actions[0] is None or actions[0] == "None":
                    print("ERROR !!!")
                return actions[0]
            
            closestFoodProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestFood)
            actions = wastarSearch(closestFoodProblem, manhattanHeuristic)
            self.updateScore(updatedGameState)
            print("Goal: " + str(closestFood))
            if closestFood in foodList:
                print("Goal Type: Food")
            elif closestFood in capsuleList:
                print("Goal Type: Capsule")
            else:
                print("Goal Type: Opponent Pacman At Home")
                print("Opponent Pacman List: " + str(opponentPacmanList))
                print("Score: " + str(score))
            if len(actions) == 0:
                print("Food list not empty, however, no action available")
                actions = self.chooseLegalRandomAction(currentPosition, wallList)
            print("Action: " + actions[0])
            print("===============================")
            print()
            print()
            updatedGameState.data.layout.walls = original_wall_grids
            gameState = updatedGameState
            if actions[0] is None or actions[0] == "None":
                print("ERROR !!!")
            return actions[0]
        
        
        elif self.mode == "invader hunting mode" and len(opponentList) != 0:
            print("Opponent List: " + str(opponentList))
            safeList = []
            for foodCoordinate in foodList:
                if self.isSafeCoordinate(foodCoordinate, updatedGameState):
                    safeList.append(foodCoordinate)
            for capsuleCoordinate in capsuleList:
                safeList.append(capsuleCoordinate)
            print("Number of Safe Goods: " + str(len(safeList)))
            if len(safeList) == 0:
                print("Mode Change: invader retreat mode")
                actions = self.retreat(updatedGameState)
                updatedGameState.data.layout.walls = original_wall_grids
                gameState = updatedGameState
                if actions[0] is None or actions[0] == "None":
                    print("ERROR !!!")
                return actions[0]
            closestSafe, distance = self.closestObject(safeList, updatedGameState)
            if closestSafe is not None:
                closestFoodProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestSafe)
                actions = wastarSearch(closestFoodProblem, manhattanHeuristic)
            else:
                print("Mode Change: invader retreat mode")
                actions = self.retreat(updatedGameState)
                print("Go Home")
                if len(actions) == 0:
                    # actions = self.chooseLegalRandomAction(currentPosition, wallList)
                    actions = self.bestAvoidGhostAction(currentPosition, wallList, self.getOpponentList(gameState))
                print("Action: " + str(actions[0]))
                if actions[0] is None or actions[0] == "None":
                    print("ERROR !!!")
                return actions[0]
            
            self.updateScore(updatedGameState)
            print("Goal: " + str(closestSafe))
            if closestSafe in foodList:
                print("Goal Type: Safe Food")
            elif closestSafe in capsuleList:
                print("Goal Type: Safe Capsule")
            else:
                print("Goal Type: Closest Home")
            if len(actions) == 0:
                # actions.append("Stop")
                print("Go Home")
                actions = self.retreat(updatedGameState)
                if len(actions) == 0:
                    print("Empty Action List, Random Select Legal Actions")
                    # actions = self.chooseLegalRandomAction(currentPosition, wallList)
                    actions = self.bestAvoidGhostAction(currentPosition, wallList, self.getOpponentList(gameState))
            print("Action: " + actions[0])
            print("===============================")
            print()
            print()
            updatedGameState.data.layout.walls = original_wall_grids
            gameState = updatedGameState
            if actions[0] is None or actions[0] == "None":
                print("ERROR !!!")
            return actions[0]
        
        
        elif self.mode == "invader power mode":
            goodList = []
            for element in foodList:
                if element not in goodList:
                    goodList.append(element)
            for element in capsuleList:
                if element not in goodList and element not in wallList:
                    goodList.append(element)
            scaredTimerCountingDownThreshold = 6 # More conservative
            scaredOpponentGhostDistance = 1
            for element in opponentList:
                for key in opponentDict:
                    if element == key[1] and element not in wallList and opponentDict[key] >= scaredTimerCountingDownThreshold and element not in goodList:
                        if self.getMazeDistance(currentPosition, element) <= scaredOpponentGhostDistance:
                            goodList.append(element)
            # huntingPacmanScoreThreshold = 5
            # score = self.getScore(updatedGameState)
            # if score > huntingPacmanScoreThreshold:
            #     for element in opponentPacmanList:
            #         if element not in foodList:
            #             foodList.append(element)
            if len(goodList) == 0:
                print("Trigger invader retreat mode")
                actions = self.retreat(updatedGameState)
                self.updateScore(updatedGameState)
                print("Goal Type: Closest Home")
                if len(actions) == 0:
                    # actions.append("Stop")
                    print("Empty Action List, Random Select Legal Actions")
                    # actions = self.chooseLegalRandomAction(currentPosition, wallList)
                    actions = self.bestAvoidGhostAction(currentPosition, wallList, self.getOpponentList(gameState))
                print("Action: " + actions[0])
                print("===============================")
                print()
                print()
                updatedGameState.data.layout.walls = original_wall_grids
                gameState = updatedGameState
                if actions[0] is None or actions[0] == "None":
                    print("ERROR !!!")
                return actions[0]
            
            closestFood, distance = self.closestObject(goodList, updatedGameState)
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
                actions = self.retreat(updatedGameState)
                if len(actions) == 0:
                    # actions = self.chooseLegalRandomAction(currentPosition, wallList)
                    actions = self.bestAvoidGhostAction(currentPosition, wallList, self.getOpponentList(gameState))
            print("Action: " + actions[0])
            print("Capsule Eaten: " + str(updatedGameState.data._capsuleEaten))
            print("Scared Timer: " + str(scaredTime))
            print("===============================")
            print()
            print()
            updatedGameState.data.layout.walls = original_wall_grids
            gameState = updatedGameState
            if actions[0] is None or actions[0] == "None":
                print("ERROR !!!")
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
                    if (homeWidth, i) not in wallList:
                        candidateHomeList.append((homeWidth, i))
            closestHome, distance = self.closestObject(candidateHomeList, updatedGameState)
            goHomeProblem = PositionSearchProblem(updatedGameState, updatedGameState.getAgentPosition(self.index), goal=closestHome)
            actions = wastarSearch(goHomeProblem, manhattanHeuristic)
            self.updateScore(updatedGameState)
            print("Goal: " + str(closestHome))
            print("Goal Type: Closest Home")
            if len(actions) == 0:
                # actions.append("Stop")
                print("Empty Action List, Random Select Legal Actions")
                actions = []
                x, y = currentPosition[0], currentPosition[1]
                if (x - 1, y) not in wallList:
                    actions.append("West")
                if (x + 1, y) not in wallList:
                    actions.append("East")
                if (x, y - 1) not in wallList:
                    actions.append("South")
                if (x, y + 1) not in wallList:
                    actions.append("North")
                if len(actions) != 0:
                    actions[0] = random.choice(actions)
                else:
                    actions.append("Stop")
            print("Action: " + actions[0])
            print("===============================")
            print()
            print()
            updatedGameState.data.layout.walls = original_wall_grids
            gameState = updatedGameState
            if actions[0] is None or actions[0] == "None":
                print("ERROR !!!")
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
            scaredTimerCountingDownThreshold = 4 # IDEA: maybe 5, which will make the invader more conservative.
            if gameState.data._capsuleEaten is None and scaredTime >= scaredTimerCountingDownThreshold:
                self.mode = "invader power mode"
                return
            if gameState.data._capsuleEaten is None and scaredTime < scaredTimerCountingDownThreshold:
                self.mode = "invader hunting mode"
            #escapeDistanceThreshold = 3
            #escapeScoreThreshold = 0
            #if closestOpponentDistance <= escapeDistanceThreshold:
            #    currentScore = self.getScore(gameState)
            #    if (currentScore > escapeScoreThreshold and self.red) or (currentScore < -1 * escapeScoreThreshold and not self.red):
            #        self.mode = "invader retreat mode"
            #    else:
            #        self.mode = "invader hunting mode"
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

    def registerInitialState(self, gameState):
        """
        This func will be called when:
            > the first time the chosen agent obj was created
            > at the beginning of each new game
        :param gameState: the initial game state
        """
        CaptureAgent.registerInitialState(self, gameState)

        # dimensions of the grid world w * h
        self.maze_dim = (gameState.data.layout.width, gameState.data.layout.height)

        self.opponentIndices = self.getOpponents(gameState)

        self.start = gameState.getAgentPosition(self.index)
        self.initialPosition = gameState.getAgentPosition(self.index)
        half_width = self.maze_dim[0] // 2
        # If the initial position is on the right side
        if self.initialPosition[0] > half_width:
            self.boarder_mid = (half_width, self.maze_dim[1] // 2)
        else:
            self.boarder_mid = (half_width - 1, self.maze_dim[1] // 2)

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
        print("invaders: ", invaders)
        if invaders:
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
        goHomeProblem = PositionSearchProblem(gameState, self.currentPosition, goal=enemyPosition)
        actions = wastarSearch(goHomeProblem, manhattanHeuristic)
        if len(actions) > 0:
            return actions[0]
        else:
            # self.defend(gameState)
            return 'Stop'

    def defend(self, gameState):
        # foodDefending, distance = self.farthestObjectUsingPosition(self.getFoodYouAreDefending(gameState).asList(),
        #                                                            self.initialPosition)
        defendFoodProblem = PositionSearchProblem(gameState, self.currentPosition, goal=self.boarder_mid)
        actions = wastarSearch(defendFoodProblem, manhattanHeuristic)
        if len(actions) > 0:
            return actions[0]
        else:
            return 'Stop'

    def flee(self, gameState, enemyPosition):
        foodDefending, distance = self.closestObjectUsingPosition(self.getFoodYouAreDefending(gameState).asList(),
                                                                  enemyPosition)
        fleeProblem = FleeProblem(gameState, self.currentPosition, enemyPosition, goal=foodDefending)
        actions = wastarSearch(fleeProblem, manhattanHeuristic)
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
        # Pick the nearest invader to hunt
        target = sorted(invaders,
                        key=lambda pos: pow(pos[0] - self.currentPosition[0], 2) + pow(pos[1] - self.currentPosition[1],
                                                                                       2))[0]

        defendFoodProblem = PositionSearchProblem(gameState, self.currentPosition, goal=target)
        actions = wastarSearch(defendFoodProblem, manhattanHeuristic)
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