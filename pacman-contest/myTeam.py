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
    # invader starting mode, invader hunting mode, invader power mode
    # defender TODO
    
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
    
    def evaluate(self, gameState, action):
        pass
    
    def closestObject(self, listOfObjects, gameState):
        currentPosition = gameState.getAgentPosition(self.index)
        closestObject = None
        closestDistance = sys.maxsize
        for candidateObject in listOfObjects:
            if self.getMazeDistance(candidateObject, currentPosition) < closestDistance:
                closestObject = candidateObject
        return closestObject, closestDistance

######################################
#            WA* Agents
######################################
class WaStarInvader(DummyAgent):
    """
    Invader Behavior design:
      1.
        * Starting mode:
         Head straight to opponent's territory (priority: High)
         Should find the shortest way to the boundary, where there are the most amount of food.
         
      2.
        * Power mode: AllFoodSearch problem
            - current score > threshold (acceptable score??) && scared_time < 30 ?? :
                    P(chase scared ghosts) >> P(eat food)
            - current score < threshold:
                  ignore scared ghosts and go straight for food
                    P(chase scared ghosts) << P(eat food)
        * Normal mode: search food & capsules, avoid ghosts
              P(avoid ghost) >> P(eat capsule) > P(eat food)
      3. Score difference:
        * myteam_score + opponent_score >= higher_threshold > 0:
                      Highly Advantaged --> Defend First
        * lower_threshold < myteam_score + opponent_score < higher_threshold:
                      Balanced --> Keep Invading
        * myteam_score + opponent_score <= lower_threshold:
                      Highly Disadvantaged --> Give-up Defending
  
    """
    
    mode = "invader starting mode"
    
    def chooseAction(self, gameState):
        if self.mode == "invader starting mode":
            closestFood, distance = self.closestObject(self.getFood(gameState).asList(), gameState)
            closestFoodProblem = PositionSearchProblem(gameState, gameState.getAgentPosition(self.index), goal=closestFood)
            actions = wastarSearch(closestFoodProblem, manhattanHeuristic)
            return actions[0]
    
    def evaluate(self, gameState, action):
        print("hi")
    
    def getFeatures(self, gameState, action):
        # features could be a list of heuristic values e.g [foodHuer, capsuleHeur, avoidGhost, huntGhost,...]
        # Possible features including:
        # Under Invader Normal Mode: getFood, getCapsule, getOpponents, getScore
        # Under Invader Power Mode: getOpponents,
        # Under Invader Retreat Mode: getBorder
        # Under Invader Starting Mode: getBorder
        foodList = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)
        opponentList = []
        for opponentIndex in self.getOpponents(gameState):
            if not gameState.getAgentState(opponentIndex).isPacman:
                opponentList.append(gameState.getAgentPosition(opponentIndex))
        closestFood, closestFoodDistance = self.closestObject(foodList, gameState)
        remainingFood = len(foodList)
        closestCapsule, closestCapsuleDistance = self.closestObject(capsuleList, gameState)
        remainingCapsule = len(capsuleList)
        notNoneOpponentList = []
        for i in range(len(opponentList)):
            if opponentList[i] is not None:
                notNoneOpponentList.append(opponentList[i])
        closestOpponent = None
        closestOpponentDistance = sys.maxsize
        if len(notNoneOpponentList) != 0:
            closestOpponent, closestOpponentDistance = self.closestObject(notNoneOpponentList, gameState)
        scoreDifference = self.getScore(gameState)
        return{"closestFoodDistance": closestFoodDistance, "remainingFood": remainingFood, "closestCapsuleDistance": closestCapsuleDistance, "remainingCapsule": remainingCapsule, "closestOpponentDistance": closestOpponentDistance, "score": scoreDifference}
    
    def getWeights(self, gameState, action):
        # Weights reflects priorities of features. The weight list varies as the game advances
        # e.g, Initially, the invader agent cares anything less than how to cross the boarder
        if self.mode == "invader starting mode":
            print("Invader Starting Mode")
            return {"crossBorder": 1, "eatOpponent": 0, "eatFood": 0, "eatCapsule": 0, "score": 0}
        elif self.mode == "invader normal mode":
            print("Invader Normal Mode")
        elif self.mode == "invader power mode":
            print("Invader Power Mode")
        elif self.mode == "invader retreat mode":
            print("Invader Retreat Mode")
        pass


class WaStarDefender(DummyAgent):
    """
      Defender Behavior design:
      1. Patrol around home area -- search and chase pacman (priority: High)
      2.
  
    """
    def chooseAction(self, gameState):
        closestFood, distance = self.closestObject(self.getFood(gameState).asList(), gameState)
        closestFoodProblem = PositionSearchProblem(gameState, gameState.getAgentPosition(self.index), goal=closestFood)
        actions = wastarSearch(closestFoodProblem, manhattanHeuristic)
        return actions[0]
    
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

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

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

