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
               first = 'InvaderConvDQN', second = 'DefenderConvDQN', **kwargs):
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
from keras.models import Sequential
from keras.layers import Input, Add, Dense, \
    Activation, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.optimizers import Adam
from keras.initializers import glorot_uniform, Constant
import numpy as np

class ConvDQN():
    def __init__(self):
        self.numTrained = 0
        self.state_size = (32, 16, 7)
        self.num_actions = 5 # [stop, north, south, west, east]
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.model_config()
        self.load()
        self.model_name = ''

    def model_config(self):
        '''
        define the model structure
        :return:  model object
        '''
        model = Sequential()
        # conv1  filters=32, kernel_size=5, stride=1, padding=valid   (?,32,16,7) -> (?, 28,12,32)
        model.add(Conv2D(32, (5, 5), strides=(1, 1), padding='valid',
                              activation='relu', bias_initializer=Constant(0.1),
                              kernel_initializer=glorot_uniform()))

        # MaxPool1    pool_size = 3, stride=1, padding=same   (28,12,32) -> (26,20,32)
        model.add(MaxPooling2D((3, 3), strides=(1, 1), padding='valid'))

        # conv2  filters=64, kernel_size=3, stride=1, padding=valid   (26,20,32) -> (24,4,64)
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid',
                              activation='relu', bias_initializer=Constant(0.1),
                              kernel_initializer=glorot_uniform()))

        # MaxPool2    pool_size = 2, stride=2, padding=same   (24,4,64) -> (24,4,64)
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        # conv3  filters=128, kernel_size=2, stride=2, padding=valid   (24,4,64) -> (12,2,128)
        model.add(Conv2D(128, (2, 2), strides=(2, 2), padding='valid',
                              activation='relu', bias_initializer=Constant(0.1),
                              kernel_initializer=glorot_uniform()))

        # AvgPool3    pool_size = 2, stride=1, padding=valid   (12,2,128) -> (11,1,128)
        model.add(AveragePooling2D((2, 2), strides=(1, 1), padding='valid'))
        model.add(Flatten())  # (12,2,128) -> (1,3072)

        # FC1 ( in: 1408   out: 256)
        model.add(Dense(256, activation='relu'))

        # FC2 ( in: 256 out: 256)
        model.add(Dense(256, activation='relu'))

        # Output layer ( in: 256  out: 5)
        model.add(Dense(self.num_actions, activation='linear'))

        # Adam Optimizer
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate) )
        return model

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
            print("\nTrying to load model...", end=' ')
            if model_file is not None:
                self.model.load_weights(model_file)
            else:
                for file in os.listdir(os.getcwd()):
                    if file.endswith(".model"):
                        model_indices.append(int(re.findall(r'_(\d+)', file)[0]))
                self.numTrained = max(model_indices)
                self.model_name = "myModel_episode_{}.model".format(self.numTrained)
                model_file = os.path.join(os.getcwd(), self.model_name)
                self.model.load_weights(model_file)

            if self.numTrained != 0:
                print("Model has been loaded !\n")
            else:
                print("No pre-trained model...\nBuild new model...\n")
        except:
            print("No pre-trained model...\nBuild new model...\n")

    def save(self, replace=True):
        self.numTrained += 1
        model_path = os.path.join(os.getcwd(), "myModel_episode_{}.model".format(self.numTrained))
        if not replace:
            self.model.save_weights(model_path)
            print("\nModel saved as myModel_episode_{}.model\n".format(self.numTrained))
        else:
            try:
                os.remove(os.path.join(os.getcwd(), self.model_name))
                self.model.save_weights(model_path)
            except:
                print("\nFail to delete previous model")
                self.model.save_weights(model_path)
            print("Model saved as myModel_episode_{}.model\n".format(self.numTrained))



class SoftmaxBody:
    def __init__(self, T):
        self.T = T  # temperature, the higher t introduces more uncertainty

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def softmax_with_temperature(self, x):
        '''
        :param x: Q_value array
        :return: the index of chosen action
        '''
        x = np.array(x) / self.T
        print("\nscale T: ", x)
        x = self.softmax(x)
        print("Probs: ", x)
        action = np.argmax(np.random.multinomial(1, x, 1))
        print("Prediction: {}, choose action {}\n".format(x, action))
        return action  # action index

from collections import deque, namedtuple
class ReplayMemory(object):
    def __init__(self, capacity=10000):
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

    def push(self, history):
        self.buffer.append(history)
        while len(self.buffer) > self.capacity: # we accumulate no more than the capacity (10000)
            self.buffer.popleft()


# define one step tuple
Step = namedtuple('Step', ['state', 'action', 'reward'])
class NStepProgress:
    """
    the agent learns from a n_step history, not for a single step
    """
    def __init__(self, ai, n_step=5):
        self.ai = ai
        self.rewards = []
        self.n_step = n_step
        self.history = deque()

    def yieldsHistory(self, oneStep):
        self.rewards.append(oneStep.reward)
        self.history.append(oneStep)
        while len(self.history) > self.n_step:
            self.history.popleft()
        if len(self.history) == self.n_step:
            return tuple(self.history)
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
        self.controller = None
        self.n_steps = None
        self.memory = None

        self.prev_state = None
        self.current_state = None

        self.prev_reward = 0
        self.step_reward = 0

        self.prev_action = None

        self.invader = None

        self.maze_dim = None
        self.walls = None
        self.wall_matrix = None

        # training
        self.epochs = None
        self.cumulated_rewards = 0.0
        self.num_of_step = 0

    def registerInitialState(self, gameState):
        self.current_state = gameState
        self.walls = gameState.getWalls()
        # dimensions of the grid world w * h
        self.maze_dim = (gameState.data.layout.width, gameState.data.layout.height)

        self.wall_matrix = self.getWallMatrix()

        conv_dqn = ConvDQN()
        softmax_body = SoftmaxBody(T=7.0)
        self.controller = AI(brain=conv_dqn, body=softmax_body)

        self.n_steps = NStepProgress(ai=self.controller, n_step=5)
        self.memory = ReplayMemory(capacity=10000)

        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        self.current_state = gameState.deepCopy()
        # Controller picks an action
        state_volume = self.getStateVolume(self.current_state)
        candidate_action_index = self.controller(state_volume)

        if self.prev_state is not None:
            self.addOneStep(self.current_state)
        else:
            self.prev_state = self.current_state.deepCopy()

        self.prev_action = candidate_action_index
        candidate_action = self.getDirection(candidate_action_index)
        legal_act = gameState.getLegalActions(self.index)
        if candidate_action not in legal_act:
            return Directions.STOP

        # print(candidate_action)
        return candidate_action

    def calculateRewardForPrevAction(self, gameState):

        util.raiseNotDefined()

    def addOneStep(self, gameState):
        self.prev_reward = self.calculateRewardForPrevAction(gameState)
        prev_step = Step(state=self.getStateVolume(self.prev_state),
                         action=self.prev_action, reward=self.prev_reward)
        prev_history = self.n_steps.yieldsHistory(oneStep=prev_step)
        if prev_history is not None:
            self.memory.push(prev_history)
        self.prev_state = gameState.deepCopy()

    def final(self, gameState):
        '''
          override function
        :param gameState:
        :return:
        '''
        # self.prev_reward = self.calculateReward()
        score = self.getScore(gameState)
        if score > 0:
            self.prev_reward = 100
        elif score < 0:
            self.prev_reward = -100
        else:
            self.prev_reward = -50

        final_step = Step(state=self.getStateVolume(self.prev_state), action=self.prev_action, reward=self.prev_reward)
        prev_history = self.n_steps.yieldsHistory(oneStep=final_step)
        if prev_history is not None:
            self.memory.push(prev_history)
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
        elif action_index == 3.: return Directions.SOUTH
        elif action_index == 4.: return Directions.WEST
        elif action_index == 5.: return Directions.EAST
        else: return Directions.STOP

    def eligibility_trace(self, batch):
        gamma = 0.99
        inputs = []
        targets = []
        for series in batch:  # series --> one history containing n steps
            input = np.array([series[0].state, series[-1].state], dtype=np.float32)
            output = self.controller.brain.forward(input)
            cumulated_reward = np.max(output[1])
            for step in reversed(series[:-1]):
                cumulated_reward = step.reward + gamma * cumulated_reward
            state = series[0].state
            target = output[0].data
            target[series[0].action] = cumulated_reward
            inputs.append(state)
            targets.append(target)
        return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)

    def train(self):
        print("=============Episode{}==============".format(self.controller.brain.numTrained+1))
        print("Start training... ReplayMemory size: {}".format(len(self.memory.buffer)))
        epochs = round(len(self.memory.buffer)/32) + 1
        for i in range(epochs):
            for batch in self.memory.sample_batch(32):  # batch size is 32
                inputs, targets = self.eligibility_trace(batch)  # input should be tensors
                # print("inputs shape: {},  targets shape: {}".format(inputs.shape, targets.shape))
                self.controller.brain.batch_learn(inputs, targets)
        step_average = np.mean(self.n_steps.rewards)
        print("Episode: %s,   Step Average Reward: %s" % (str(i), str(step_average)))
        self.controller.brain.save(replace=True)


class InvaderConvDQN(BasicAgentConvDQN):
    """
    reward plan:
        1. eat a food dot:  +1
        2. eat a capsule: + 0.8
        3. kill a scared ghost: +0.4
        4. win a game: +100
        5. eaten by a ghost: -1.2
        6. eaten an opponents +0.8
        7. lose a game: -100
    """
    def calculateRewardForPrevAction(self, gameState):
        return np.random.randint(-20, 20)


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

