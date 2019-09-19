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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
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
    """
    Picks among actions randomly.
    This part implements the chosen algorithm acutually  -- tech 1: heuristic search wastar
    """
    # chosen from ['North', 'South', 'West', 'East', 'Stop']
    actions = gameState.getLegalActions(self.index)

    for action in actions:
      succ = self.getSuccessor(gameState, action)

      # evaluate the one successor, as this is not a path-finding task, it should be
      # "real-time", thus the algorithm should determine which is the best move among
      # five possible actions

    '''
    You should change this in your own agent.
    '''
    return random.choice(actions)

  def getSuccessor(self, gameState, action):

    successor = gameState.generateSuccessor(self.index, action)
    position = successor.getAgentState(self.index).getPosition()
    if position != util.nearestPoint(position):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor


  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

######################################
#            WA* Agents
######################################
class WaStarInvader(DummyAgent):
  """
  Invader Behavior design:
    1. Head straight to opponent's territory (priority: High)
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
  def chooseAction(self, gameState):
    pass

  def getFeatures(self, gameState, action):
    # features could be a list of heuristic values e.g [foodHuer, capsuleHeur, avoidGhost, huntGhost,...]
    pass

  def getWeights(self, gameState, action):
    # Weights reflects priorities of features. The weight list varies as the game advances
    # e.g, Initially, the invader agent cares anything less than how to cross the boarder
    pass


class WaStarDefender(DummyAgent):
  """
    Defender Behavior design:
    1. Patrol around home area -- search and chase pacman (priority: High)
    2.

  """
  def chooseAction(self, gameState):
    pass

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
class PositionSearchProblem:
  pass
