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
    def __init__(self, state_size, num_actions):
        self.state_size = state_size
        self.num_actions = num_actions
        self.memory = ReplayMemory(capacity=100000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.model_config()

    def model_config(self):
        '''
        define the model structure
        :return:  model object
        '''
        model = Sequential()
        # conv1
        model.add(Conv2D(32, (5, 5), strides=(1, 1), padding='same',
                              activation='relu', bias_initializer=Constant(0.1),
                              kernel_initializer=glorot_uniform()))
        model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

        # conv2
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid',
                              activation='relu', bias_initializer=Constant(0.1),
                              kernel_initializer=glorot_uniform()))
        model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

        # conv3
        model.add(Conv2D(128, (2, 2), strides=(1, 1), padding='valid',
                              activation='relu', bias_initializer=Constant(0.1),
                              kernel_initializer=glorot_uniform()))
        model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        model.add(Flatten())

        # FC1 ( in: flatten units   out: 256)
        model.add(Dense(256, input_dim=self.count_neurons(self.state_size), activation='relu'))

        # FC2 ( in: 256 out: 256)
        model.add(Dense(256, activation='relu'))

        # Output layer ( in: 256  out: num_actions)
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

    def learn(self, input, target):
        self.model.train_on_batch(input, target)

    # TODO
    def count_neurons(self, input_size):
        return 0

    # def learn(self, samples):
    #     for state, next_state, reward, action in samples:
    #         Q_target = reward + self.gamma * np.amax(self.model.predict(next_state))
    #         target = predict = self.model.predict(state)
    #         target[0][action] = Q_target
    #         self.model.fit(state, target, epochs=1, verbose=0)
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
    #
    # def update(self, reward, new_state):
    #     self.memory.push( (self.last_state, new_state, self.last_action, self.last_reward) )
    #     action = self.choose_action(new_state)
    #     if len(self.memory.backup) > 32:
    #         self.learn(self.memory.sampling(32))
    #     self.last_action = action
    #     self.last_state = new_state
    #     self.last_reward = reward
    #     return action

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    # def choose_action(self, state):
    #     Q_values = self.model.predict(state)
    #     softmax_out = self.softmax_with_temperature(Q_values, 7)
    #     # action = np.where(softmax_out[0] == 1)[0][0]
    #     action = self.softmax_with_temperature(Q_values, 7)
    #     return action

class SoftmaxBody:
    def __init__(self, T):
        self.T = T  # temperature, the higher t introduces more uncertainty

    def softmax_with_temperature(self, x):
        '''
        :param x: Q_value array
        :return: the index of chosen action
        '''
        x = np.array(x) ** (1 / self.T)
        x_sum = x.sum()
        x = x / x_sum
        return np.argmax(np.random.multinomial(1, x, 1))  # action index

from collections import deque, namedtuple
class ReplayMemory(object):
    def __init__(self, n_steps, capacity=10000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()

    def sample_batch(self, batch_size):  # creates an iterator that returns random batches
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs+1)*batch_size <= len(self.buffer):
            yield vals[ofs*batch_size: (ofs+1)*batch_size]
            ofs += 1

    def run_steps(self, samples):
        while samples > 0:
            entry = next(self.n_steps_iter) # 10 consecutive steps
            self.buffer.append(entry) # we put 200 for the current episode
            samples -= 1
        while len(self.buffer) > self.capacity: # we accumulate no more than the capacity (10000)
            self.buffer.popleft()

# define one step tuple
Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])
class NStepProgress:
    """
    the agent learns from a n_step history, not for a single step
    """
    def __init__(self, pacman_interface, ai, n_step=5):
        self.ai = ai
        self.pacman_interface = pacman_interface
        self.rewards = []
        self.n_step = n_step

    def __iter__(self):
        state = self.pacman_interface.reset()
        history = deque()
        reward = 0.0
        while True:
            action = self.ai(np.array([state]))[0][0]
            next_state, r, is_done, _ = self.pacman_interface.step(action)

            reward += r
            history.append(Step(state=state, action=action, reward=r, done=is_done))
            while len(history) > self.n_step + 1:
                history.popleft()
            if len(history) == self.n_step + 1:
                yield tuple(history)

            state = next_state
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()

                self.rewards.append(reward)
                reward = 0.0
                state = self.pacman_interface.reset()
                history.clear()

    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps


class AI:
    def __init__(self, brain, body):
        self.brain = brain  # ConvDQN
        self.body = body  # SoftmaxBody

    def __call__(self, input_state):
        input = tf.Variable(np.array(input_state, dtype=np.float32))
        output = self.brain.forward(input)
        actions = self.body.softmax_with_temperature(output)
        return actions.data.numpy()


class MoveAverage:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    def average(self):
        return np.mean(self.list_of_rewards)

# training
# state "image" pre-processing

# TODO
def state_resize(input_state):
    return

input_image = None
state_size = state_resize(input_image)
num_actions = 5

# create a pacman game envrionment
pacman_interface = None

# Build an AI
cnn = ConvDQN(state_size, num_actions)
softmax_body = SoftmaxBody(T=2.0)
ai = AI(brain=cnn, body=softmax_body)
ma = MoveAverage(500)

# Set up experience replay
n_steps = NStepProgress(pacman_env, ai=ai, n_step=5)
memory = ReplayMemory(n_steps=n_steps, capacity=10000)


def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch: # series --> one history containing n steps
        input = np.array([series[0].state, series[-1].state], dtype=np.float32)
        output = ai.brain.forward(input)
        cumulated_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumulated_reward = step.reward + gamma * cumulated_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumulated_reward
        inputs.append(state)
        targets.append(target)
    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)

# Training the AI
# loss = MSELoss()
# optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
nb_epochs = 100

for epoch in range(1, nb_epochs + 1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128): # batch size is 128
        inputs, targets = eligibility_trace(batch)   # input should be numpy array
        # predictions = cnn.forward(inputs)
        # loss_error = loss(predictions, targets)
        # optimizer.zero_grad()
        # loss_error.backward()
        # optimizer.step()
        ai.brain.learn(inputs, targets)
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
    # if avg_reward >= 1500:
    #     print("Congratulations, your AI wins")
    #     break


    # test
    win_rate = 0.7

    ai.brain.save("model_Epoch_{}_winrate_{}.h5".format(nb_epochs, win_rate))



########################
#     ConvDQN Agents
########################
class BasicAgentConvDQN(CaptureAgent):
    def __init__(self):
        super(BasicAgentConvDQN, self).__init__()

        conv_dqn = ConvDQN(state_size, num_actions)
        softmax_body = SoftmaxBody(T=2.0)
        self.controller = AI(brain=conv_dqn, body=softmax_body)

        self. wall_matrix = self.getWallMatrix()

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)

        for action in actions:
            succ = self.getSuccessor(gameState, action)


    # baseline matrix = shape of the maze
    # state model: (wallMaxtrix, agentPos, foodMatrix)
     # TODO
    def getWallMatrix(self):
        return None

    def getFoodMatrix(self, gameState):
        pass

    def getCapsuleMatrix(self, gameState):

    def getOpponentMatrix(self, gameState):
        # Observable opponents -- > ghost: Scared: 1   normal:-1    pacman: 1
        pass

    def getMyPositionMatrix(self, gameState):
        pass





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
    pass

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
    pass

