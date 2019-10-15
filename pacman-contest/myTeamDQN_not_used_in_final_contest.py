# myTeamDQN_not_used_in_final_contest.py
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
               first = 'MixedConvDQN', second = 'MixedConvDQN', **kwargs):
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

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    # print(bestActions)

    return random.choice(bestActions)


  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)  #self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}



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
    def __init__(self, index, model_file=None):
        self.numTrained = 0
        self.state_size = (34, 18, 1)  # The RANDOM maze is of size (34, 18)
        self.num_actions = 4  # [north, south, west, east]
        self.gamma = 0.9    # discount rate
        self.learning_rate = 0.001
        self.model = Sequential()
        self.model_name = ''
        self.index = index
        self.save_interval = 1
        self.last_save_num = 0
        self.load(model_file)

    def model_config(self):
        '''
        define the model structure
        :return:  model object
        '''
        # conv1  filters=32, kernel_size=5, stride=1, padding=valid   (?,34,18,1) -> (?, 30,14,32)
        self.model.add(Conv2D(32, (5, 5), strides=(1, 1), padding='valid', input_shape=self.state_size))

        # # MaxPool1    pool_size = 3, stride=1, padding=same
        # model.add(MaxPooling2D((3, 3), strides=(1, 1), padding='valid'))

        # conv2  filters=64, kernel_size=3, stride=1, padding=valid   (?, 30,14,128) -> (?, 26,10,64)
        self.model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='valid'))

        # # MaxPool2    pool_size = 2, stride=2, padding=same
        # model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        # paras: bias_initializer=Constant(0.1), kernel_initializer=glorot_uniform()
        # conv3  filters=128, kernel_size=3, stride=1, padding=valid   (?, 26,10,256) -> (?, 12,4,128)
        self.model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='valid'))

        # # AvgPool3    pool_size = 2, stride=1, padding=valid
        # model.add(AveragePooling2D((2, 2), strides=(1, 1), padding='valid'))

        self.model.add(Flatten())  # (?, 12,4,128) -> (?, 1,6144)

        # FC ( in: 6144   out: 256)
        self.model.add(Dense(256, activation='relu'))

        # Output layer ( in: 256  out: 4)
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
        batch_loss = self.model.train_on_batch(input, target)
        return batch_loss

    def load(self, model_file=None):
        try:
            model_indices = [0]
            print("\nTrying to load model...{}".format(model_file if model_file else ''), end=' ')
            if model_file is not None:
                self.model.load_weights(model_file)
            else:
                for file in os.listdir(os.getcwd()):
                    if file.endswith(".model"):
                        if 'Model_{}_'.format(self.index) in file:
                            model_indices.append(int(re.findall(r'_(\d+)', file)[0]))
                self.numTrained = max(model_indices)
                self.model_name = "Model_{}_episode_{}.model".format(self.index, self.numTrained)
                model_file = os.path.join(os.getcwd(), self.model_name)
            if self.numTrained != 0:
                self.model = load_model(model_file)
                print("MODEL: {} has been loaded !\n".format(self.model_name))
                self.last_save_num = self.numTrained
            else:
                print("No pre-trained model...\n Build new model...\n")
                self.model_config()
            self.last_save_num = self.numTrained
        except:
            print("No pre-trained model...\n Build new model...\n")
            self.model_config()

    def save(self):
        if self.numTrained == self.last_save_num + self.save_interval:
            model_name = "Model_{}_episode_{}.model".format(self.index, self.numTrained)
            model_path = os.path.join(os.getcwd(), model_name)
            save_model(self.model, filepath=model_path)
            self.last_save_num = self.numTrained

        # remove earlier models
        try:
            model_index_to_remove = []
            for file in os.listdir(os.getcwd()):
                if file.endswith(".model"):
                    if 'Model_{}_'.format(self.index) in file:
                        model_index_to_remove.append(int(re.findall(r'_(\d+)', file)[0]))
            if len(model_index_to_remove) > 10:
                model_index_to_remove = sorted(model_index_to_remove)[:-3]
                for index in model_index_to_remove:
                    remove_name = "Model_{}_episode_{}.model".format(self.index, index)
                    os.remove(os.path.join(os.getcwd(), remove_name))
        except:
            print("\nError occurred when trying to delete earlier models")
        print("Model saved as {}\n".format(model_name))


class SoftmaxBody:
    def __init__(self, T):
        self.T = T  # temperature, the higher t introduces more uncertainty

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / sum(np.exp(x))

    def softmax_with_temperature(self, x):
        '''
        :param x: Q_value array
        :return: the index of chosen action
        '''
        print("Q Values: ", x)
        x = np.asarray(x).astype(np.float64) / self.T
        x = self.softmax(x)
        # action = np.argmax(np.random.multinomial(1, x, 1))
        action = np.argmax(x)
        print("Probability: {}, Action chosen: {}\n".format(x, action))
        return action  # action index


from collections import deque, namedtuple
class ReplayMemory(object):
    def __init__(self, capacity=100000):  # Stores up to 100000 (by default) n-step move histories for training
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
        '''
            Everytime a previous step is added, it genrates a series of continuous steps as
            histories
        '''

        self.history.append(oneStep)
        while len(self.history) > self.n_step + 1:
            self.history.popleft()
        if len(self.history) == self.n_step + 1:
            return [tuple(self.history)]
        if oneStep.isDone:
            history_list = []
            if len(self.history) > self.n_step + 1:
                self.history.popleft()
            while len(self.history) >= 1:
                history_list.append(tuple(self.history))
                self.history.popleft()
            self.history.clear() # clear if one game episode ends
            return history_list
        return None


class AI:
    '''
        The core of our DQN agents.
        It has two parts:
            > 1. Convolutional DQN Network --> acts as agent brain
            > 2. Softmaxbody  --> take the outputs of brain for a given input volume, and decide what the next action is
    '''
    def __init__(self, brain, body):
        self.brain = brain  # ConvDQN
        self.body = body  # SoftmaxBody

    def __call__(self, input_state):
        input_state = np.array(input_state, dtype=np.float64)
        input_volume = self.check_dim(input_state)
        input_volume = input_volume.reshape((1,) + input_volume.shape)

        output = self.brain.forward(input_volume)[0]
        # print("output: {},  length: {}, dtype: {}".format(output, len(output), type(output[0])))

        actions = self.body.softmax_with_temperature(output)
        return actions

    def check_dim(self, input_state):
        if input_state.shape != (34, 18, 1):  # RAMDOM Maze size
            exit(print("Error! The Maze should be RANDOM Mazes of size (34, 18) !\n"
                       "Please use RAMDOM maze \n"
                       "> Try: python3 capture.py -r <Team> -b <Team> -l RANDOM<seed>"))

        return input_state


########################
#     ConvDQN Agents
########################
class BasicAgentConvDQN(CaptureAgent):
    """
    Basic agent class of both InvaderConvDQN and DefenderConvDQN
    """
    def __init__(self, index):
        super().__init__(index, timeForComputing=.1 )
        # Initialize AI
        self.controller = AI(brain=ConvDQN(index=index), body=SoftmaxBody(T=2.0))
        self.n_steps = NStepProgress(n_step=5)

        # to store a large number of step move histories, this huge memory history is used when a game episode ends
        self.memory = ReplayMemory(capacity=100000)

        # to store limited number of step move histories, this tiny memory history is used after each action was decided
        self.tiny_memory = ReplayMemory(capacity=32)

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
        self.prev_score = 0
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
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):

        # update local copy of the current state and team scores
        self.current_state = gameState.deepCopy()
        self.current_score = self.getScore(gameState)

        # Controller picks an action
        print("Step {} Forward:".format(self.num_of_step))
        state_volume = self.getStateVolume(self.current_state)
        candidate_action_index = self.controller(state_volume)
        candidate_action_index = int(candidate_action_index)

        if self.prev_state is not None:
            # not the first move
            self.addOneStep(self.current_state)
        else:
            # first move of a game
            self.prev_state = self.current_state.deepCopy()

        # Mapping: action index -- > action str
        candidate_action = self.getDirection(candidate_action_index)
        legal_act = gameState.getLegalActions(self.index)

        self.prev_action = candidate_action_index  # record the action made regardless of legality
        self.legal_prev_action = candidate_action in legal_act
        if candidate_action not in legal_act:
            return Directions.STOP  # illegal choice of action leads to 'STOP'
        else:
            return candidate_action  # valid action

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
        self.cumulated_rewards += self.prev_reward  # update cumulated_rewards of current game episode

        # create a Step obj of previous game state
        prev_step = Step(state=self.getStateVolume(self.prev_state),
                         action=self.prev_action, reward=self.prev_reward, isDone=self.gameOver)
        print("Step {} Feedback:".format(self.num_of_step-1))
        print("Position: {}, Action: {}, Reward: {}\n".format(self.prev_state.getAgentState(self.index).getPosition(),
                                                            self.getDirection(prev_step.action),
                                                            prev_step.reward))
        histoies = self.n_steps.yieldsHistory(oneStep=prev_step)
        if histoies:
            self.memory.push(histoies)
            # self.tiny_memory.push(histoies)

        # uncomment this to do update network after each move
        # for batch in self.tiny_memory.sample_batch(4):  # batch size is 4
        #     inputs, targets = self.eligibility_trace(batch)
        #     # print("inputs shape: {},  targets shape: {}".format(inputs.shape, targets.shape))
        #     self.controller.brain.batch_learn(inputs, targets)  # learn from tiny memory

        # update state and score,  when stats from previous move recorded
        self.prev_state = gameState.deepCopy()
        self.prev_score = self.current_score
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
        self.addOneStep(gameState)  # record last step of a game
        self.local_episode_count += 1
        self.train()  # train on huge memory
        self.initializeGameStats()  # reset game related parameters

    def getStateVolume(self, state):
        '''
            Old Scheme:
                stack all the matices
                state model: (w * h * num_of_channels), in this case, num_of_channels = 7
                (wallMaxtrix, foodMatrix, capsuleMatrix, myPositionMatrix, pacmanMatrix, GhostMatrix, ScaredGhostMatrix)
            New Scheme:
                state model: (w * h * 1), with all aspects encoded with specified values.
                This scheme aims to simulate a agent vision with a  grey scale "picture" representation
                of the gameState.

        :return: an input volume of shape (w,h, 7 or 1) for the network
        '''

        # old scheme
        # matrices = []
        # matrices.append(self.wall_matrix)
        # matrices.append(self.getFoodMatrix(state))
        # matrices.append(self.getCapsuleMatrix(state))
        # matrices.append(self.getMyPositionMatrix(state))
        # matrices.append(self.getPacmanMatrix(state))
        # matrices.append(self.getGhostMatrix(state))
        # matrices.append(self.getScaredGhost(state))
        # state_volume = np.stack(matrices, axis=2)

        # New scheme
        state_volume = self.wall_matrix
        state_volume = np.add(state_volume, self.getFoodMatrix(state))
        state_volume = np.add(state_volume, self.getCapsuleMatrix(state))
        state_volume = np.add(state_volume, self.getMyPositionMatrix(state))
        state_volume = np.add(state_volume, self.getPacmanMatrix(state))
        state_volume = np.add(state_volume, self.getGhostMatrix(state))
        state_volume = np.add(state_volume, self.getScaredGhost(state))

        # self.matrixViewer(state_volume/255)
        state_volume = np.reshape(state_volume, (self.maze_dim[0], self.maze_dim[1], 1))
        state_volume = state_volume / 255.  # normalize
        assert state_volume.shape == (self.maze_dim[0], self.maze_dim[1], 1)
        assert state_volume.any() <= 1, 'value error'
        return state_volume

    def getWallMatrix(self):
        '''
            Old scheme: isWall: 0, not wall: 1
            New scheme: isWall: 0,  not wall: 255
        '''
        wallMatrix = np.zeros(self.maze_dim, dtype=np.float64)
        return self.matrixValueMapping1(self.walls, wallMatrix, map_dict={"True":0, "False":255})

    def getFoodMatrix(self, gameState):
        '''
            Old scheme: myFood: -1, targetFood: 1, all others: 0
            New scheme: myFood: 20, targetFood: 30
        '''
        foodMatrix = np.zeros(self.maze_dim, dtype=np.float64)
        defendFoodMatrix = np.zeros(self.maze_dim, dtype=np.float64)
        Food = self.getFood(gameState)
        defendFood = self.getFoodYouAreDefending(gameState)
        foodMatrix = self.matrixValueMapping1(Food, foodMatrix, map_dict={"True": 30, "False": 0})
        defendFoodMatrix = self.matrixValueMapping1(defendFood, defendFoodMatrix, map_dict={"True": 20, "False": 0})

        return np.add(foodMatrix, defendFoodMatrix)

    def getCapsuleMatrix(self, gameState):
        '''
            Old scheme: myCapsule: -1,  opponent capsule: 1, all others: 0
            New scheme: myCapsule: 25,  opponent capsule: 35
        '''
        capsuleMatrix = np.zeros(self.maze_dim, dtype=np.float64)
        capsule_list = gameState.getCapsules()
        if len(capsule_list) > 0:
            for (x, y) in capsule_list:
                if gameState.isOnRedTeam(self.index):
                    capsuleMatrix[x][-1-y] = 25 if x <= self.maze_dim[0] // 2 else 35
                else:
                    capsuleMatrix[x][-1-y] = 35 if x <= self.maze_dim[0] // 2 else 25
        return capsuleMatrix

    def getGhostMatrix(self, gameState):
        '''
            Old scheme: Observable Ghosts -- > myTeam ghost: -1   opponent ghost: 1, all others: 0
            New scheme: Observable Ghosts -- > myTeam ghost: 40   opponent ghost: 50
        '''
        matrix = np.zeros(self.maze_dim, dtype=np.float64)
        find_position = lambda index: gameState.getAgentPosition(index) \
            if not gameState.getAgentState(index).isPacman \
                and not gameState.getAgentState(index).scaredTimer > 0 \
                    else (-1, -1)
        self.matrixValueMapping2(matrix, gameState, find_position, (40, 50))
        return matrix

    def getPacmanMatrix(self, gameState):
        '''
            Old scheme: myTeam pacman: 60,  opponent pacman: 70
            New scheme: myTeam pacman: -1,  opponent pacman: 1, all others: 0
        '''
        matrix = np.zeros(self.maze_dim, dtype=np.float64)
        find_position = lambda index: gameState.getAgentPosition(index) \
            if gameState.getAgentState(index).isPacman else (-1, -1)
        # print("from getPacmanMatrix: ")
        self.matrixValueMapping2(matrix, gameState, find_position, (60, 70))
        return matrix

    def getScaredGhost(self, gameState):
        '''
            Old scheme: scared myTeam Ghost: -1,   scared opponent Ghost: 1, all others: 0
            New scheme: scared myTeam Ghost: 45,   scared opponent Ghost: 55
        '''
        matrix = np.zeros(self.maze_dim, dtype=np.float64)
        find_position = lambda index: gameState.getAgentPosition(index) \
            if not gameState.getAgentState(index).isPacman \
                and gameState.getAgentState(index).scaredTimer > 0 \
                    else (-1, -1)
        self.matrixValueMapping2(matrix, gameState, find_position, (45, 55))
        return matrix

    def getMyPositionMatrix(self, gameState):
        '''
            Old scheme: myPosition: 1, all others: 0
            New scheme: myPosition: 100
        '''
        myPositionMatrix = np.zeros(self.maze_dim, dtype=np.float64)
        x,y = gameState.getAgentPosition(self.index)
        myPositionMatrix[x][-1-y] = 100
        return myPositionMatrix

    def matrixViewer(self, matrix):
        '''
        Nice print of a matrix
        '''
        for i in range(self.maze_dim[1]):
            print(matrix.T[i])

    def matrixValueMapping1(self, valueGrid, matrix, map_dict):
        w, h = self.maze_dim
        for i in range(w):
            for j in range(h):
                matrix[i][-1 - j] = map_dict["True"] if valueGrid[i][j] else map_dict["False"]
        return matrix

    def matrixValueMapping2(self, matrix, gameState, func, values):
        # values = (1, -1)
        if not gameState.isOnRedTeam(self.index):
            values = values[::-1]
        for pos in map(func, gameState.getRedTeamIndices()):
            if pos and pos[0] + pos[1] > 0: matrix[pos[0]][-1-pos[1]] = values[0]
        for pos in map(func, gameState.getBlueTeamIndices()):
            if pos and pos[0] + pos[1] > 0: matrix[pos[0]][-1-pos[1]] = values[1]
        return matrix

    def getActionIndex(self, direction):
        '''
            Map from action name to corresponding id
        '''
        if direction == Directions.NORTH: return 0
        elif direction == Directions.SOUTH: return 1
        elif direction == Directions.WEST: return 2
        elif direction == Directions.EAST: return 3
        else: return None

    def getDirection(self, action_index):
        '''
            Map from action id to action name
        '''
        if action_index == 0: return Directions.NORTH
        elif action_index == 1: return Directions.SOUTH
        elif action_index == 2: return Directions.WEST
        elif action_index == 3: return Directions.EAST
        else: return None

    def eligibility_trace(self, batch):
        '''
            The agent learns from a given number of steps, which make up a history, instead of learning from the feedback
            of one single step move.
            The rewards of steps are accumulated from the very last one to the first, applying a decay factor.
            This function basically expects a better estimation of how good or bad a step move accounts for the next
            steps across the N-step history (series).
        '''
        gamma = 0.9  # reward decay
        inputs = []
        targets = []
        for series in batch:  # series --> one history containing n steps
            input = np.array([series[0].state, series[-1].state], dtype=np.float32)
            output = self.controller.brain.forward(input)  # Q-value prediction
            cumulated_reward = 0.0 if series[-1].isDone else np.max(output[1])
            for step in reversed(series[:-1]):
                cumulated_reward = step.reward + gamma * cumulated_reward
            state = series[0].state
            target = output[0].data
            target[series[0].action] = cumulated_reward
            inputs.append(state)  # batch input volumes
            targets.append(target)  # batch rewards (Q-prediction)
        return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)

    def train(self):
        if self.memory.__len__() > 16:
            print("========================Episode {}========================".format(self.controller.brain.numTrained))
            print("Local Episode: {},   Reward: {},   Step Reward Average: {}".format(str(self.local_episode_count+1), str(self.cumulated_rewards), str(self.cumulated_rewards / self.num_of_step)))
            print("Start training...                  ReplayMemory size: {} ".format(len(self.memory.buffer)))
            loss_list = []
            for batch in self.memory.sample_batch(16):  # batch size is 16
                inputs, targets = self.eligibility_trace(batch)  # input should be tensors
                # print("inputs shape: {},  targets shape: {}".format(inputs.shape, targets.shape))
                loss_list.append(self.controller.brain.batch_learn(inputs, targets))
            print("Mean Batch loss: {}".format(np.mean(loss_list)))
            self.controller.brain.numTrained += 1
            self.controller.brain.save()
            print("==========================================================")


REWARD_BASE = 1
class MixedConvDQN(BasicAgentConvDQN):
    """
    rewards plan:
        1. eat a food dot:  20
        2. eat a capsule: 25
        3. kill a scared ghost: -2
        4. win a game: 100
        5. eaten by a ghost: -50
        6. kill an opponents: 20
        7. lose a game: -100
        8. living bonus or penalty: 1 or 5 as pacman
        9. legal action: 1 or -1 if illegal
    """

    def getFeatures(self, gameState):
        features = {}

        prevAgentState = self.prev_state.getAgentState(self.index)
        currAgentState = gameState.getAgentState(self.index)

        myPosition = currAgentState.getPosition()
        prevMyPosition = prevAgentState.getPosition()

        isPacmanPrev = prevAgentState.isPacman
        isPacmanNow = currAgentState.isPacman

        boarder_x = self.maze_dim[0] // 2

        def legalActionReward():
            return 1 if self.legal_prev_action else -1

        def capsuleFoodReward(agent):
            myCapsules = agent.prev_state.data.capsules
            if prevMyPosition:
                x, y = int(prevMyPosition[0]), int(prevMyPosition[1])
            else:
                return 0
            if isPacmanPrev:
                if myCapsules and prevMyPosition in myCapsules:
                    return 25
                # elif agent.prev_state.data.food[x][y]:
                elif prevMyPosition == self.prev_state.data._foodEaten:
                    return 20
            return 0

        def travelReward(agent):
            delta_x = 0
            if myPosition and prevMyPosition:
                x1, _ = prevMyPosition
                x2, _ = myPosition
                delta_x = x2 - x1
            if myPosition == prevMyPosition: return -2
            if not isPacmanPrev:  # at home previously
                if agent.prev_action == 0 or agent.prev_action == 1:  # vertical move
                    return 0
                else:  # horizontal move
                    # Action: 0,1,2,3 -> N,S,W,E
                    if agent.red:  # red agent
                        if agent.prev_action == 3:  # move to enemy
                            return 0 if delta_x <= 0 else 2
                        elif agent.prev_action == 2:  # move home
                            return 0
                    else:  # blue agent
                        if agent.prev_action == 4:  # move home
                            return 0
                        elif agent.prev_action == 3:  # move to enemy
                            return 0 if delta_x >= 0 else 2

            else: # at enemy's territory previously
                if agent.red:
                    if agent.prev_action == 3:  # move home
                        return -1 if prevAgentState.numCarrying < 2 else 2
                else:
                    if agent.prev_action == 4:  # move to enemy
                        return -1 if prevAgentState.numCarrying < 2 else 2
            return 0

        def scaredReward(agent):
            """
              Get rid of scared mode: 5
            :param agent:
            :return:
            """
            if prevAgentState.scaredTimer > 0:
                scaredTimerChange = abs(currAgentState.scaredTimer - prevAgentState.scaredTimer)
                if currAgentState.scaredTimer < prevAgentState.scaredTimer:
                    return 0
                else:
                    if scaredTimerChange > 1:
                        return 5
            return 0

        def deathReward(agent):
            """
             Died as pacman: -50
             Died as scared ghost: 20
            :param agent:
            :return:
            """
            if isPacmanPrev:  # isPacman
                # if manhattanDistance(myPosition, prevMyPosition) > 1:
                if not prevMyPosition:
                    return -50
            else: # isGhost ,  gain 20 if previously scared and got killed now
                scaredTimerChange = abs(currAgentState.scaredTimer - prevAgentState.scaredTimer)
                if scaredTimerChange > 1:
                    if prevAgentState.scaredTimer > 0:
                        return 20  # bonus, as the agent can be back to normal asap
            return 0

        def killReward(agent):
            """
                Kill pacman: 20  huge reward
                Kill scared ghost: -2  slight penalty
            :param agent:
            :return:
            """
            def manhattanDistance(p1, p2):
                return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

            if prevMyPosition:
                if not isPacmanPrev:  # is ghost  previously
                    for index in agent.getOpponents(self.prev_state):
                        otherAgentState = self.prev_state.data.agentStates[index]
                        if not otherAgentState.isPacman: continue
                        pacPos = otherAgentState.getPosition()
                        if pacPos == None: continue
                        if manhattanDistance(pacPos, prevMyPosition) <= 0.7:
                            if prevAgentState.scaredTimer <= 0:  # kill enemy pacman
                                return 20

                else:  # is pacman previously
                    for index in agent.getOpponents(self.prev_state):
                        otherAgentState = self.prev_state.data.agentStates[index]
                        if otherAgentState.isPacman: continue
                        pacPos = otherAgentState.getPosition()
                        if pacPos == None: continue
                        if manhattanDistance(pacPos, prevMyPosition) <= 0.7:
                            if otherAgentState.scaredTimer > 0:  # kill an scared ghost
                                return -5
            return 0

        def lossReward(agent):
            return self.current_score - self.prev_score

        def gameWinReward(agent):
            if gameState.data._win:
                print("Game end, I'm red?  {},  my score is: {}".format(agent.red, gameState.getScore() if self.red else -1*gameState.getScore()))
                if gameState.getScore() == 0:
                    return 0
                elif gameState.getScore() > 0:
                    return 100 if agent.red else -100
                elif gameState.getScore() < 0:
                    return -100 if agent.red else 100
            return 0

        def livingReward(agent):
            boarder_dist_bias = abs(boarder_x - prevMyPosition[0]) if prevMyPosition else boarder_x
            step_bias = 0 if agent.prev_score >= 0 else self.num_of_step
            if isPacmanPrev:
                # return 1+0.2*(np.sqrt(self.num_of_step + boarder_dist_bias))
                return 5
            else:
                # return -1-0.2*(np.sqrt(self.num_of_step + boarder_dist_bias))
                return 1

        # calculate defined feature values
        features["CapsuleFood"] = capsuleFoodReward(self)
        features["Travel"] = travelReward(self)
        features["ScaredMode"] = scaredReward(self)
        features["Loss"] = lossReward(self)
        features["KillEnemy"] = killReward(self)
        features["Death"] = deathReward(self)
        features["LegalAction"] = legalActionReward()
        features["GameWin"] = gameWinReward(self)
        features["LivingReward"] = livingReward(self)

        return features

    def getWeights(self, gameState):
        weights = {
            "CapsuleFood": 1,
            "Travel": 0,
            "ScaredMode": 0,
            "Loss": 0,
            "KillEnemy": 1,
            "Death": 1,
            "LivingReward": 1,
            "LegalAction": 1,
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
