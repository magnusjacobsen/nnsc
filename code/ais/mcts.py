import numpy as np
import math
import hex.game as game
import time
import random

class Config():
  def __init__(self, num_iterations, exploration_parameter):
    self.actors = 1 # assuming only one GPU is available

    self.num_iterations = num_iterations # the number of iterations we do for each PUCT run

    # UCB formula values
    #self.prior_constant_base = 19652 # copied from AlphaZero
    #self.prior_constant_init = 1.25 # copied from AlphaZero
    self.exploration_parameter = exploration_parameter # with 800 iterations, exploration parameter would start at 1.25 and end a about 1.29 (at the root level)

class Node():
  def __init__(self, state, prior = 0.5, action = None, parent = None):
    self.state = state
    self.prior = prior # TODO: should be changed later
    self.t = 0
    self.n = 0
    self.parent = parent # parent node, used in back propagation
    self.action = action # action that lead us here
    self.children = []
    self.untried_actions = game.actions(state)
    random.shuffle(self.untried_actions) # to remove bias

  def update(self, result):
    self.n += 1
    self.t += result

class AI():
  def __init__(self, num_iterations = 800, exploration_parameter = 0.635):
    self.config = Config(num_iterations, exploration_parameter)
    self.version = 'MCTS' + str(num_iterations)
    self.epoch = ''

  def decide_move(self, state):
    root = Node(state)
    suggested = mcts(root, self.config)
    stop = time.time()
    return suggested.action

  def decide_parallel(self, states):
    actions = [self.decide_move(state) for state in states]
    return actions

  def __str__(self):
    return f'mcts (simulations: {self.config.num_iterations}, exploration parameter: {self.config.exploration_parameter})'
  def __repr__(self):
    return 'mcts.AI()'


# TODO: maybe the key needs to be slightly addjusted, to also take Q into account?
def mcts(root, config):
  for _ in range(config.num_iterations):
    # select
    node = select(root, config)
    #expand
    node = expand(node)
    # evaluate
    state = evaluate(node.state)
    # backpropagate
    backpropagate(node, state)
  return max(root.children, key = lambda c : c.n) # return the child with the most visits

def select(node, config):
  while not node.untried_actions and node.children:
    node = max(node.children, key = lambda c : ucb(c, config))
  return node

def expand(node):
  if node.untried_actions:
    action = node.untried_actions.pop()
    child = Node(game.result(node.state, action), action = action, parent = node)
    node.children.append(child)
    node = child
  return node

def evaluate(state):
  while not game.terminal_test(state):
    actions = game.actions(state)
    rand_action = actions[random.randrange(0, len(actions))]
    state = game.result(state, rand_action)
  return state

def backpropagate(node, state):
  while node:
    node.update(game.utility(state, -game.player(node.state)))
    node = node.parent

def ucb(node, config):  
  exploration_score = config.exploration_parameter * (math.sqrt(node.parent.n) / (node.n + 1))
  return (node.t / node.n) + exploration_score

def pp_node(node, config):
  p = f'p: {game.player(node.state)}, '
  a = f'a: {node.action:.03d}, '
  avg = f'avg_val: {node.t / node.n}, '
  ucb_score = f'ucb: {ucb(node, config)}, '
  visits = f'visits: {node.n}/{node.parent.n}, '
  print(p + a + avg + ucb_score + visits)