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
from util import nearestPoint
import re, os


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'InvaderConvDQN', second = 'IdleAgent', **kwargs):
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


class IdleAgent(CaptureAgent):
    """An agent that does not move"""
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
    def chooseAction(self,gameState):
        return Directions.STOP


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

#
# def heuristic_switcher(state, gameState):
#   pass

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
    # state model: ((x,y), )
  pass





########################################
#          Deep Q-Learning
########################################
import tensorflow as tf
from keras.models import Sequential, load_model, save_model
from keras.layers import Input, Add, Dense, \
    Activation, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.optimizers import Adam
from keras.initializers import glorot_uniform, Constant
import numpy as np


class ConvDQN():
    def __init__(self, model_file=None):
        self.numTrained = 0
        self.state_size = (32, 16, 7)
        self.num_actions = 5 # [stop, north, south, west, east]
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.002
        self.model = Sequential()
        self.model_name = ''

        self.load(model_file)

    def model_config(self):
        '''
        define the model structure
        :return:  model object
        '''
        # conv1  filters=32, kernel_size=5, stride=1, padding=valid   (?,32,16,7) -> (?, 28,12,32)
        self.model.add(Conv2D(32, (5, 5), strides=(1, 1), padding='valid', activation='relu'))

        # # MaxPool1    pool_size = 3, stride=1, padding=same
        # model.add(MaxPooling2D((3, 3), strides=(1, 1), padding='valid'))

        # conv2  filters=64, kernel_size=3, stride=1, padding=valid   (?, 28,12,32) -> (?, 24,8,64)
        self.model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu'))

        # # MaxPool2    pool_size = 2, stride=2, padding=same
        # model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        # paras: bias_initializer=Constant(0.1), kernel_initializer=glorot_uniform()
        # conv3  filters=128, kernel_size=3, stride=1, padding=valid   (?, 24,8,64) -> (?, 20,4,128)
        self.model.add(Conv2D(128, (5, 5), strides=(1, 1), padding='valid', activation='relu'))

        # # AvgPool3    pool_size = 2, stride=1, padding=valid
        # model.add(AveragePooling2D((2, 2), strides=(1, 1), padding='valid'))

        self.model.add(Flatten())  # (20,4,128) -> (?, 1,10240)

        # FC1 ( in: 10240   out: 256)
        self.model.add(Dense(256, activation='relu'))

        # FC2 ( in: 256 out: 256)
        self.model.add(Dense(256, activation='relu'))

        # Output layer ( in: 256  out: 5)
        self.model.add(Dense(self.num_actions, activation='linear'))

        # Adam Optimizer
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate) )

    def forward(self, state):
        """
        :param state:  state matrices
        :return: Q_value prediction
        """
        return self.model.predict(state)

    def batch_learn(self, input, target):
        self.model.train_on_batch(input, target)

    # TODO
    def count_neurons(self, input_size):
        return 0

    def load(self, model_file=None):
        try:
            model_indices = [0]
            print("\nTrying to load model...{}".format(model_file if model_file else ''), end=' ')
            if model_file is not None:
                self.model.load_weights(model_file)
            else:
                for file in os.listdir(os.getcwd()):
                    if file.endswith(".model"):
                        model_indices.append(int(re.findall(r'_(\d+)', file)[0]))
                self.numTrained = max(model_indices)
                self.model_name = "myModel_episode_{}.model".format(self.numTrained)
                model_file = os.path.join(os.getcwd(), self.model_name)
                self.model = load_model(model_file)
            if self.numTrained != 0:
                print("MODEL: {} has been loaded !\n".format(self.model_name))
            else:
                print("No pre-trained model...\n Build new model...\n")
                self.model_config()
        except:
            print("No pre-trained model...\n Build new model...\n")
            self.model_config()

    def save(self):
        model_path = os.path.join(os.getcwd(), "myModel_episode_{}.model".format(self.numTrained))
        save_model(self.model, filepath=model_path)

        # remove earlier models
        try:
            model_index_to_remove = []
            for file in os.listdir(os.getcwd()):
                if file.endswith(".model"):
                    model_index_to_remove.append(int(re.findall(r'_(\d+)', file)[0]))
            if len(model_index_to_remove) > 10:
                model_index_to_remove = sorted(model_index_to_remove)[:-3]
                for index in model_index_to_remove:
                    os.remove(os.path.join(os.getcwd(), "myModel_episode_{}.model".format(index)))
        except:
            print("\nError occurred when trying to delete earlier models")
        print("Model saved as myModel_episode_{}.model\n".format(self.numTrained))


class SoftmaxBody:
    def __init__(self, T):
        self.T = T  # temperature, the higher t introduces more uncertainty

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / float(sum(np.exp(x)))

    def softmax_with_temperature(self, x):
        '''
        :param x: Q_value array
        :return: the index of chosen action
        '''
        print("linear: ", x)
        x = np.asarray(x).astype('float64') / self.T
        x = self.softmax(x)
        # print("Probs: ", x)
        # action = np.argwhere(x == np.amax(x))[0][0]

        # original version
        # x = np.asarray(x).astype('float64') ** (1 / self.T)
        # x_sum = x.sum()
        # x = x / x_sum

        print(x)
        print(np.random.multinomial(1, x, 1))
        action = np.argmax(np.random.multinomial(1, x, 1))
        print("Prediction: {}, choose action {}\n".format(x, action))
        return action  # action index

from collections import deque, namedtuple
class ReplayMemory(object):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque()

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):  # creates an iterator that returns random batches
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs+1)*batch_size <= len(self.buffer):
            yield vals[ofs*batch_size: (ofs+1)*batch_size]
            ofs += 1

    def push(self, histories):
        # we accumulate no more than the capacity (100000)
        while self.__len__() + len(histories) > self.capacity:
            self.buffer.popleft()
        for history in histories:
            self.buffer.append(history)


# define one step tuple
Step = namedtuple('Step', ['state', 'action', 'reward', 'isDone'])
class NStepProgress:
    """
    the agent learns from a n_step history, not for a single step
    """
    def __init__(self, n_step=5):
        self.n_step = n_step
        self.history = deque()

    def yieldsHistory(self, oneStep):

        self.history.append(oneStep)
        while len(self.history) > self.n_step:
            self.history.popleft()
        if len(self.history) == self.n_step:
            return [tuple(self.history)]
        if oneStep.isDone:
            history_list = []
            if len(self.history) > self.n_step:
                self.history.popleft()
            while len(self.history) >= 1:
                history_list.append(tuple(self.history))
                self.history.popleft()
            self.history.clear() # clear if one game episode ends
            return history_list
        return None


class AI:
    def __init__(self, brain, body):
        self.brain = brain  # ConvDQN
        self.body = body  # SoftmaxBody

    def __call__(self, input_state):
        input_state = np.array(input_state, dtype=np.float32)
        input_volume = self.check_dim(input_state)
        input_volume = input_volume.reshape((1,) + input_volume.shape)

        output = self.brain.forward(input_volume)[0]
        # print("output: {},  length: {}, dtype: {}".format(output, len(output), type(output[0])))

        actions = self.body.softmax_with_temperature(output)
        return actions

    # TODO
    def check_dim(self, input_state):
        if input_state.shape != (32, 16, 7):
            # resize the input
            pass
        return input_state

    # TODO
    def resize_input(self, input_state):
        pass


########################
#     ConvDQN Agents
########################
class BasicAgentConvDQN(CaptureAgent):
    def __init__(self, index, timeForComputing = .1):
        super().__init__(index, timeForComputing)
        # Initialize AI
        self.controller = AI(brain=ConvDQN(), body=SoftmaxBody(T=10))
        self.n_steps = NStepProgress(n_step=5)
        self.memory = ReplayMemory(capacity=10000)

        # One game episode stats
        self.maze_dim = None
        self.walls = None
        self.wall_matrix = None
        self.invader = None

        # training paras
        self.numTraining = 0    # number of training episodes
        self.local_episode_count = 0

    def initializeGameStats(self):
        # game-related stats
        self.prevScore = 0
        self.current_score = 0
        self.legal_prev_action = False
        self.prev_state = None
        self.current_state = None
        self.prev_reward = 0
        self.step_reward = 0
        self.prev_action = None
        self.cumulated_rewards = 0.0
        self.num_of_step = 0
        self.gameOver = False


    def registerInitialState(self, gameState):

        self.current_state = gameState
        self.walls = gameState.getWalls()
        # dimensions of the grid world w * h
        self.maze_dim = (gameState.data.layout.width, gameState.data.layout.height)
        self.wall_matrix = self.getWallMatrix()

        self.initializeGameStats()
        self.start = gameState.getAgentPosition(self.index)
        self.gameOver = False
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        self.current_state = gameState.deepCopy()
        self.current_score = self.getScore(gameState)

        # Controller picks an action
        state_volume = self.getStateVolume(self.current_state)
        candidate_action_index = self.controller(state_volume)

        if self.prev_state is not None:
            # not the first move
            self.addOneStep(self.current_state)
        else:
            # first move of a game
            self.prev_state = self.current_state.deepCopy()

        # Mapping: action index -- > action str
        candidate_action = self.getDirection(candidate_action_index)
        print("Action {}: {}".format(candidate_action_index, candidate_action))
        legal_act = gameState.getLegalActions(self.index)
        self.prev_action = candidate_action_index # record the action made regardless of legality
        # self.legal_prev_action = candidate_action in legal_act
        if candidate_action not in legal_act:
            random_choose_legal = np.random.choice(legal_act, 1)
            self.prev_action = int(self.getActionIndex(random_choose_legal))
            return random_choose_legal[0] # if not legal, returns stop instead, but introduces penalty
        return candidate_action

    def calculateRewardForPrevAction(self, gameState):

        util.raiseNotDefined()

    def addOneStep(self, gameState):
        """
        To calculate the Q_value of an prev_action in prev_gameState, we
        need to know the reward gained from prev_action. To calculate the
        reward, currentState and prevState are needed. So, self.prev_state
        and prev_Score shouldn't get updated before prev_reward being calculated.

        1. Get the reward for previous move --> self.prev_reward
        2. Create a Step object and push into NStepProgress history
        3. prev_gameState and prev_score can be updated to currentState
           and currentScore when Step pushed into NStepProgress
        :param gameState: current state of the game
        :return:
        """
        self.prev_reward = self.calculateRewardForPrevAction(gameState)
        self.cumulated_rewards += self.prev_reward # update cumulated_rewards of current game episode

        # create a Step obj of previous game state
        prev_step = Step(state=self.getStateVolume(self.prev_state),
                         action=self.prev_action, reward=self.prev_reward, isDone=self.gameOver)
        print("Step {}:".format(self.num_of_step+1))
        print("Position: {}, Action: {}, Reward: {}".format(self.prev_state.getAgentState(self.index).getPosition(),
                                                            self.getDirection(prev_step.action),
                                                            prev_step.reward))
        histoies = self.n_steps.yieldsHistory(oneStep=prev_step)
        if histoies:
            self.memory.push(histoies)

        # update state and score,  when stats from previous move recorded
        self.prev_state = gameState.deepCopy()
        self.prevScore = self.current_score
        self.num_of_step += 1


    def final(self, gameState):
        '''
         This func will be excuted when a game should be terminated.
         After resetting game related states get updated, the agent
         expects a training with steps histories in his memory before
         moving to next game episode.

         Note: that the terminal state will not be able to generate any
         successor state.

        :param gameState: terminal state
        '''
        self.gameOver = True

        self.addOneStep(gameState) # record last step of a game
        self.train()


    # state model: (w * h * num_of_channels), in this case, num_of_channels = 7
    #   (wallMaxtrix, foodMatrix, capsuleMatrix, myPositionMatrix, pacmanMatrix, GhostMatrix, ScaredGhostMatrix)

    def getWallMatrix(self):
        '''
         isWall: 1,  not wall: 0
        '''
        wallMatrix = np.zeros(self.maze_dim, dtype=np.int)
        return self.matrixValueMapping1(self.walls, wallMatrix, map_dict={"True":1, "False":0})

    def getFoodMatrix(self, gameState):
        '''
        myFood: -1, targetFood: 1, all others: 0
        '''
        foodMatrix = np.zeros(self.maze_dim, dtype=np.int)
        defendFoodMatrix = np.zeros(self.maze_dim, dtype=np.int)
        Food = self.getFood(gameState)
        defendFood = self.getFoodYouAreDefending(gameState)
        foodMatrix = self.matrixValueMapping1(Food, foodMatrix, map_dict={"True": 1, "False": 0})
        defendFoodMatrix = self.matrixValueMapping1(defendFood, defendFoodMatrix, map_dict={"True": -1, "False": 0})
        # if gameState.isOnRedTeam(self.index):
        #     redFoodMatrix = self.matrixValueMapping(redFood, redFoodMatrix, map_dict={"True": 1, "False": 0})
        #     blueFoodMatrix = self.matrixValueMapping(blueFood, blueFoodMatrix, map_dict={"True": -1, "False": 0})
        # else:
        #     blueFoodMatrix = self.matrixValueMapping(blueFood, blueFoodMatrix, map_dict={"True": 1, "False": 0})
        #     redFoodMatrix = self.matrixValueMapping(redFood, redFoodMatrix, map_dict={"True": -1, "False": 0})

        return np.add(foodMatrix, defendFoodMatrix)

    def getCapsuleMatrix(self, gameState):
        '''
        myCapsule: 1,  opponent capsule: -1 , all others: 0
        '''
        capsuleMatrix = np.zeros(self.maze_dim, dtype=np.int)

        # defendCapsuleMatrix = np.zeros(self.maze_dim, dtype=np.int)
        # capsules = self.getCapsules(gameState)
        # defendCapsules = self.getCapsulesYouAreDefending(gameState)

        capsule_list = gameState.getCapsules()
        if len(capsule_list) > 0:
            for (x, y) in capsule_list:
                if gameState.isOnRedTeam(self.index):
                    capsuleMatrix[x][-1-y] = 1 if x <= self.maze_dim[0] // 2 else -1
                else:
                    capsuleMatrix[x][-1-y] = -1 if x <= self.maze_dim[0] // 2 else 1
        return capsuleMatrix

    def getGhostMatrix(self, gameState):
        '''
        Observable Ghosts -- > myTeam ghost: 1   opponent ghost: -1,  all others 0
        '''
        matrix = np.zeros(self.maze_dim, dtype=np.int)
        find_position = lambda index: gameState.getAgentPosition(index) \
            if not gameState.getAgentState(index).isPacman \
                and not gameState.getAgentState(index).scaredTimer > 0 \
                    else (-1, -1)
        self.matrixValueMapping2(matrix, gameState, find_position)
        return matrix

    def getPacmanMatrix(self, gameState):
        '''
        myTeam pacman: 1,  opponent pacman: -1,  all others : 0
        '''
        matrix = np.zeros(self.maze_dim, dtype=np.int)
        find_position = lambda index: gameState.getAgentPosition(index) \
            if gameState.getAgentState(index).isPacman else (-1, -1)
        # print("from getPacmanMatrix: ")
        self.matrixValueMapping2(matrix, gameState, find_position)

        return matrix

    def getScaredGhost(self, gameState):
        '''
        scared myTeam Ghost: 1,   scared opponent Ghost: -1,  all others: 0
        '''
        matrix = np.zeros(self.maze_dim, dtype=np.int)
        find_position = lambda index: gameState.getAgentPosition(index) \
            if not gameState.getAgentState(index).isPacman \
                and gameState.getAgentState(index).scaredTimer > 0 \
                    else (-1, -1)
        self.matrixValueMapping2(matrix, gameState, find_position)

        return matrix

    def getMyPositionMatrix(self, gameState):
        '''
        myPosition: 1,  all other: 0
        '''
        myPositionMatrix = np.zeros(self.maze_dim, dtype=np.int)
        x,y = gameState.getAgentPosition(self.index)
        myPositionMatrix[x][-1-y] = 1
        return myPositionMatrix

    def matrixViewer(self, matrix):
        for i in range(self.maze_dim[1]):
            print(matrix.T[i])

    def matrixValueMapping1(self, valueGrid, matrix, map_dict):
        w, h = self.maze_dim
        for i in range(w):
            for j in range(h):
                matrix[i][-1 - j] = map_dict["True"] if valueGrid[i][j] else map_dict["False"]
        return matrix

    def matrixValueMapping2(self, matrix, gameState, func):
        values = (1, -1)
        if not gameState.isOnRedTeam(self.index):
            values = values[::-1]
        for pos in map(func, gameState.getRedTeamIndices()):
            if pos and pos[0] + pos[1] > 0: matrix[pos[0]][-1-pos[1]] = values[0]
        for pos in map(func, gameState.getBlueTeamIndices()):
            if pos and pos[0] + pos[1] > 0: matrix[pos[0]][-1-pos[1]] = values[1]
        return matrix

    def getStateVolume(self, state):
        '''
        stack all the matices
        :return: an input volume w*h*7
        '''
        matrices = []

        matrices.append(self.wall_matrix)
        matrices.append(self.getFoodMatrix(state))
        matrices.append(self.getCapsuleMatrix(state))
        matrices.append(self.getMyPositionMatrix(state))
        matrices.append(self.getPacmanMatrix(state))
        matrices.append(self.getGhostMatrix(state))
        matrices.append(self.getScaredGhost(state))

        state_volume = np.stack(matrices, axis=2)
        assert state_volume.shape == (self.maze_dim[0], self.maze_dim[1], 7)
        return state_volume

    def getActionIndex(self, direction):
        if direction == Directions.STOP: return 0.
        elif direction == Directions.NORTH: return 1.
        elif direction == Directions.SOUTH: return 2.
        elif direction == Directions.WEST: return 3.
        elif direction == Directions.EAST: return 4.
        else: return 0

    def getDirection(self, action_index):
        if action_index == 0.: return Directions.STOP
        elif action_index == 1.: return Directions.NORTH
        elif action_index == 2.: return Directions.SOUTH
        elif action_index == 3.: return Directions.WEST
        elif action_index == 4.: return Directions.EAST
        else: return Directions.STOP

    def eligibility_trace(self, batch):
        gamma = 0.99
        inputs = []
        targets = []
        for series in batch:  # series --> one history containing n steps
            input = np.array([series[0].state, series[-1].state], dtype=np.float32)
            output = self.controller.brain.forward(input)
            cumulated_reward = 0.0 if series[-1].isDone else np.max(output[1])
            for step in reversed(series[:-1]):
                cumulated_reward = step.reward + gamma * cumulated_reward
            state = series[0].state
            target = output[0].data
            target[series[0].action] = cumulated_reward
            inputs.append(state)
            targets.append(target)
        return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)

    def train(self):
        if self.memory.__len__() > 300:
            print("===================Episode {}===================".format(self.local_episode_count+1))
            print("Local Episode: %s,   Reward: %s" % (str(self.local_episode_count+1), str(self.cumulated_rewards)))
            print("Start training... ReplayMemory size: {} ...".format(len(self.memory.buffer)), end='')
            for batch in self.memory.sample_batch(8):  # batch size is 8
                inputs, targets = self.eligibility_trace(batch)  # input should be tensors
                # print("inputs shape: {},  targets shape: {}".format(inputs.shape, targets.shape))
                self.controller.brain.batch_learn(inputs, targets)
            self.local_episode_count += 1
            self.controller.brain.numTrained += 1
            self.controller.brain.save()


REWARD_BASE = 10
class InvaderConvDQN(BasicAgentConvDQN):
    """
    reward weights plan:
        1. eat a food dot:  2
        2. eat a capsule: 5
        3. kill a scared ghost: ?
        4. win a game: 100
        5. eaten by a ghost: weighted (food loss)
        6. eaten an opponents: ?
        7. lose a game: -100
        8. Intention to cross boarder: distance change of an action
        9. cross boarder: 3
        10. living bonus or penalty: 0.2
    """
    def getFeatures(self, gameState):
        features = {}

        prevAgentState = self.prev_state.getAgentState(self.index)
        currAgentState = gameState.getAgentState(self.index)

        myPosition = currAgentState.getPosition()
        prevMyPosition = prevAgentState.getPosition()

        isPacmanPrev = prevAgentState.isPacman
        isPacmanNow = currAgentState.isPacman

        boader = self.maze_dim[0] // 2

        inHomeTerritory = myPosition[0] < boader if self.red else myPosition >= boader
        live_reward_base = 0.2
        reward = REWARD_BASE

        # def findMinimumBoarderDist(agent, pos):
        #     dists = []
        #     for y in range(1, self.maze_dim[1] - 1):
        #         if not self.walls[boader][y]:
        #             dists.append(agent.getMazeDistance(pos1=pos, pos2=(boader, y)))
        #     return min(dists)

        # def legalActionReward():
        #     return 1 if self.legal_prev_action else -3

        def capsuleFoodReward(agent):
            myCapsules = agent.prev_state.data.capsules
            if prevMyPosition:
                x, y = prevMyPosition
            else:
                return 0
            live_reward = live_reward_base
            if isPacmanPrev:
                if prevMyPosition in myCapsules:
                    return 5 + live_reward if isPacmanNow else 5 - live_reward
                elif agent.prev_state.data.food[x][y]:
                    return 2 + live_reward if isPacmanNow else 5 - live_reward
            return 0

        def tryCrossBoarderReward(agent):
            live_reward = live_reward_base
            if not isPacmanPrev:  # at home previously
                if myPosition == prevMyPosition:  # stay still
                    return -2 - live_reward
                else:
                    # Action: 0,1,2,3,4 -> STOP,N,S,W,E
                    if agent.red:  # red agent
                        if agent.prev_action == 4:  # move to enemy
                            return 1 - live_reward
                        elif agent.prev_action == 3:  # move home
                            return -3 - live_reward
                        else:
                            return -live_reward_base  # North and South
                    else:  # blue agent
                        if agent.prev_action == 4:  # move home
                            return -3-live_reward
                        elif agent.prev_action == 3:  # move enemy
                            return 1-live_reward
                        else:
                            return -live_reward_base  # North and South
            return 0

        def scaredReward(agent):
            """
              Stay scared or just back to normal: -2
              Get rid of scared mode: 5
            :param agent:
            :return:
            """
            if prevAgentState.scaredTimer > 0:
                scaredTimerChange = abs(currAgentState.scaredTimer - prevAgentState.scaredTimer)
                if currAgentState.scaredTimer < prevAgentState.scaredTimer:
                    return -2
                else:
                    if scaredTimerChange > 1:
                        return 5
            return 0

        def deathReward(agent):
            """
             Died as pacman: -5
             Died as scared ghost: 5
            :param agent:
            :return:
            """

            if isPacmanPrev: # isPacman
                # if manhattanDistance(myPosition, prevMyPosition) > 1:
                if not prevMyPosition:
                    return -5
            else: # isGhost ,  gain 5 if previously scared and got killed now
                scaredTimerChange = abs(currAgentState.scaredTimer - prevAgentState.scaredTimer)
                if scaredTimerChange > 1:
                    if prevAgentState.scaredTimer > 0:
                        return 5  # bonus, as the agent can be back to normal asap
            return 0

        def killReward(agent):
            """
                Kill pacman: 20  huge reward
                Kill scared ghost: -2  slight penalty
            :param agent:
            :return:
            """
            if prevMyPosition:
                if not isPacmanPrev:  # is ghost  previously
                    for index in agent.getOpponents(self.prev_state):
                        otherAgentState = self.prev_state.data.agentStates[index]
                        if not otherAgentState.isPacman: continue
                        pacPos = otherAgentState.getPosition()
                        if pacPos == None: continue
                        if manhattanDistance(pacPos, prevMyPosition) <= 0.7:
                            if prevAgentState.scaredTimer <= 0: # kill enemy pacman
                                return 20
                else: # is pacman previously
                    for index in agent.getOpponents(self.prev_state):
                        otherAgentState = self.prev_state.data.agentStates[index]
                        if otherAgentState.isPacman: continue
                        pacPos = otherAgentState.getPosition()
                        if pacPos == None: continue
                        if manhattanDistance(pacPos, prevMyPosition) <= 0.7:
                            if otherAgentState.scaredTimer > 0: # kill an scared ghost
                                return -2
            return 0

        def backHomeReward(agent):
            """
                > case 1: Actively move back home ( Carry food back )
                                attracts penalty if team's currently disadvantaged
                > case 2: Passively back home (Geot killed)
                                attracts huge penalty and even more penalty if team's disadvantaged
            :param agent:
            :return:
            """

            if isPacmanPrev and not isPacmanNow:
                lossGain = abs(currAgentState.numCarryin - prevAgentState.numCarrying)
                if not prevMyPosition:  # Got killed
                    alpha = -1.5 * (lossGain - live_reward_base) \
                        if agent.current_score < 0 else -(lossGain + live_reward_base)
                    return -10 + alpha
                else:
                    alpha = 1.5 * (lossGain - live_reward_base) \
                        if agent.current_score < 0 else (lossGain + live_reward_base)
                    return 1 + alpha
            return 0

        def gameWinReward(agent):
            if gameState.data._win:
                if agent.current_score > 0:
                    return 100
                elif agent.current_score == 0:
                    return -20
                elif agent.current_score < 0:
                    return -100
            return 0

        # calculate defined feature values
        features["CapsuleFood"] = capsuleFoodReward(self)
        features["CrossBoarder"] = tryCrossBoarderReward(self)
        features["ScaredMode"] = scaredReward(self)
        features["BackHome"] = backHomeReward(self)
        features["KillEnemy"] = killReward(self)
        features["Death"] = deathReward(self)
        # features["LegalAction"] = legalActionReward()
        features["GameWin"] = gameWinReward(self)

        return features

    def getWeights(self, gameState):
        weights = {
            "CapsuleFood": 0.8,
            "CrossBoarder": 0.9,
            "ScaredMode": 0.3,
            "BackHome": 0.7,
            "KillEnemy": 0.1,
            "Death": 0.7,
            # "LegalAction": 1,
            "GameWin": 1
        }
        return weights

    def evaluate(self, gameState):
        """
        Computes a linear combination of features and feature weights
        """
        sum_reward = 0.0
        features = self.getFeatures(gameState)
        weights = self.getWeights(gameState)
        for k, v in features.items():
            sum_reward += weights[k] * v

        return sum_reward * REWARD_BASE

    def calculateRewardForPrevAction(self, gameState):
        return self.evaluate(gameState)


class DefenderConvDQN(BasicAgentConvDQN):
    """
    reward plan:
        1. eat a food dot:  +0.3
        2. eat a capsule: + 0.2
        3. kill a scared ghost: +0.4
        4. kill an opponent pacman: +1
        5. win a game: +100
        6. eaten by a ghost: -2
        7. eaten an opponents +0.8
        8. lose a game: -100
    """
    def calculateRewardForPrevAction(self, gameState):
        return np.random.randint(-20, 20)

