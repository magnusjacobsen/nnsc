# Neural Network Sequence Calculator, a AlphaZero lite algorithm
# - investigating if a similar approach can reach interesting results, with only consumer-grade hardware
import math, time, sys, torch, pickle, random
import hex.gui as gui
import hex.game as game
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import nnsc.gamestorage as gs
import nnsc.network as net
import nnsc.config as cf
import nnsc.savefile as sf

# various stuff
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEBUG = True
CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

plt.ion()

def clear_up(count):
  while count:
    sys.stdout.write(CURSOR_UP_ONE)
    sys.stdout.write(ERASE_LINE)
    count -= 1

################################
#### Game tree node class ######
################################
# tree search node. Represents a game state, and an action getting us here
class Node():
  def __init__(self, prior, config, state = None, action = None, parent = None, noise = 0.0, stats = None):
    self.prior = prior
    self.expl_const = config.exploration_constant
    self.state = state
    self.action = action
    self.parent = parent
    self.noise = noise

    self.t = 0
    self.n = 0
    self.children = {} # includes all the actions
    self.value = 0
    self.prior_expl = prior
    self.parent_expl = self.expl_const

    self.stats = stats

  def update(self, result):
    self.n += 1
    self.t += result
    self.value = self.t / self.n
    self.prior_expl = self.prior / (self.n + 1)
    self.parent_expl = self.expl_const * math.sqrt(self.n)

  def save_stats(self, num_positions):
    if not self.stats: 
      self.stats = []
    sum_n = sum(child.n for child in self.children.values())
    sum_n = sum_n if sum_n else 1
    # I need to flip the root.value, since it is from the perspective of the opposing player
    q_and_visits = (-self.value, [self.children[action].n / sum_n if action in self.children else 0 for action in range(num_positions)])
    self.stats.append(q_and_visits)

# uses latest network, to make a simple MCTS recommendation for best action to take
# returna an action
# does not add noise, or use softmax sampling
class AI():
  def __init__(self, savefile):
    self.config = savefile.config
    if savefile.config.use_half_precision:
      self.network = net.copy16(savefile.network, self.config)
    else:
      self.network = savefile.network
    self.version = savefile.version
    self.epoch = savefile.num_epochs_trained

  def decide_move(self, state):
    root = Node(0, self.config, state = state)
    evaluate([root], self.network, self.config)
    for i in range(self.config.num_iterations):
      node = select(root, self.config)
      values = evaluate([node], self.network, self.config)
      backpropagate(node, values[0])
    best_node = max(root.children.values(), key = lambda c : c.n)
    return best_node.action

  def decide_parallel(self, states):
    root_nodes = [Node(0, self.config, state = s) for s in states]
    evaluate(root_nodes, self.network, self.config)
    for i in range(self.config.num_iterations):
      nodes = [select(r, self.config) for r in root_nodes]
      values = evaluate(nodes, self.network, self.config)
      for node, value in zip(nodes, values):
        backpropagate(node, value)
    return [(max(r.children.values(), key = lambda c : c.n)).action for r in root_nodes]

  def __str__(self):
    return f'Neural Network Sequence Calculator (version: {self.version}, epoch: {self.epoch})'
  def __repr__(self):
    return f'Neural Network Sequence Calculator (version: {self.version}, epoch: {self.epoch})'

####### End of classes #########
################################

def run_nnsc(savefile):
  pp_setup(savefile)

  config = savefile.config
  game_storage = savefile.game_storage
  for i in range(1, config.max_runs + 1):
    draw_plots(savefile)
    start_time = time.time()

    if savefile.save_mode:    
      if i % config.runs_between_saves == 1:
        print('saving savefile to disk')
        sf.save_to_disk(savefile)
    
    # half precision copy of network
    # only used when creating games
    if DEBUG: print('playing games')
    if config.use_half_precision:
      network_half = net.copy16(savefile.network, config)
      play_games_in_batch(i * config.num_actors,
                          config.num_actors,
                          network_half,
                          config,
                          game_storage)
    else:
      play_games_in_batch(i * config.num_actors,
                    config.num_actors,
                    savefile.network,
                    config,
                    game_storage)

    if DEBUG: print('training')
    train_network(savefile.network, config, game_storage, savefile)

    # update the savefile to current data
    finish_time = time.time()
    savefile.update(finish_time - start_time)

def play_games_in_batch(num_games_played, num_actors, network, config, game_storage):
  start = time.time()
  root_nodes = [Node(0, config, state = game.initial_state(config.size)) for _ in range(num_actors)]

  games_finished = 0
  if DEBUG:
    i = 1
    print()
    print()

  while root_nodes:
    if DEBUG:
      clear_up(2)
      print(f'turn: {i}, actors: {len(root_nodes)}/{config.num_actors}')
      i += 1
    
    suggestions = tree_search(root_nodes, config, network)
    temp = []
    for suggestion, root in zip(suggestions, root_nodes):
      state = game.result(root.state, suggestion.action)
      root.save_stats(config.action_size)
      if game.terminal_test(state):
        game_storage.save_game(gs.SavedGame(state, root, config))
        games_finished += 1
        if DEBUG: 
          winner = 1 if game.utility(state, 1) == 1 else 2
      else: 
        temp.append(Node(0, config, state = state, stats = root.stats))
    root_nodes = temp

  if DEBUG:
    end = time.time()
    t = end - start
    clear_up(2)
    print(f'{num_actors} games played in {t:.4f}/{num_actors} = {(t / num_actors):.4f}s per game.')

def tree_search(roots, config, network):
  evaluate(roots, network, config)
  for r in roots:
    add_noise(r, config)

  if DEBUG:
    old_perc = -1
    print()
  for i in range(config.num_iterations):
    if DEBUG:
      perc = int(((i + 1) / config.num_iterations) * 100)
      if perc > old_perc:
        hash = int(0.4 * perc)
        rem = 40 - hash
        clear_up(1)
        print(('▓' * hash) + ('░' * rem) + ' ' + str(perc) + '%')
        old_perc = perc

    nodes = [select(root, config) for root in roots]
    values = evaluate(nodes, network, config)
    for node, value in zip(nodes, values):
      backpropagate(node, value)
  return [select_action(r, config) for r in roots]

def select_action(root, config):
  if len(root.state.history) < config.num_softmax_sampling_moves:
    return softmax_sample(root)
  else:
    return max(root.children.values(), key = lambda c : c.n) # return the child with the most visits

def add_noise(node : Node, config):
  noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(node.children))
  noise = noise / sum(noise)
  for (key, c_node), n in zip(node.children.items(), noise):
    c_node.prior = (c_node.prior * (1 - config.root_noise_exploration_fraction) +
                    n * config.root_noise_exploration_fraction)
    c_node.noise = n

def select(node : Node, config):
  node = max(node.children.values(), key = lambda c : ucb(c)) # otherwise we would return root in first iteration
  node.state = game.result(node.parent.state, node.action)
  while node.n > 0 and not game.terminal_test(node.state):
    node = max(node.children.values(), key = lambda c : ucb(c)) # we need to unpack the action:child_node pair
    node.state = game.result(node.parent.state, node.action)
  return node

def evaluate(nodes, network, config):
  image_stacks = [[game._image_stack(n.state.history, config.size, -1) for n in nodes]]
  images = net.make_tensors(image_stacks, config)

  players = [game.player(n.state) for n in nodes]
  values, logits = net.inference(network,
                                 images,
                                 players, # man kunne måske nøjes med en enkelt spiller, hvis vi ved det altid er den samme... men det er det self ikke for alle spil
                                 config.size)
  # expanding nodes
  # essentially performing a SoftMax operation on the legal nodes
  policies = [{a: math.exp(logits[i][a]) for a in game.actions(n.state)} for i, n in enumerate(nodes)]
  for node, policy in zip(nodes, policies):
    p_sum = sum(policy.values())
    lst = [Node(p / p_sum, config, action = a, parent = node) for a, p in policy.items()]
    random.shuffle(lst) # to avoid bias for smaller values, when doing initial ucb, when prior value is the same
    for c in lst:
      node.children[c.action] = c
  if config.use_utility: # use utility function when a state is a terminal state
    values = [-v.item() if not game.terminal_test(n.state) else -(game.utility(n.state, p)) for v, p, n in zip(values, players, nodes)]
  else: # az9
    values = [-v.item() for v in values]
  return values # invert to reflect opposing player

def backpropagate(node : Node, value : int):
  if node:
    node.update(value)
    backpropagate(node.parent, -value)

def ucb(node):
  return (node.parent.parent_expl * node.prior_expl) + node.value

# using exp-normalize trick, to avoid getting 0 values
# to also work on lists, we iterate over the collection, instead of using numpy overloaded element-wise operators
# does not return same collection object, in other words, operations are not in place
@nb.jit
def safer_softmax1D(values, alpha = 1):
  #values = values.copy()
  max_v = max(values)
  sum_exp = 0 
  for i, x in enumerate(values):
    v = math.exp((x - max_v) * alpha)
    sum_exp += v
    values[i] = v
  for i, x in enumerate(values):
    values[i] = x / sum_exp
  return values

def softmax_sample(node):
  children = [c for c in node.children.values()]
  sum_visits = sum(c.n for c in children)
  # make all visits sum to 1
  visits = [c.n / sum_visits for c in children]

  visits = safer_softmax1D(visits, alpha = 1.5)
  choice = np.random.choice(children, p = visits)
  return choice

logsoftmax = nn.LogSoftmax(dim = 1).to(DEVICE)
def softmax_crossentropy_with_logits(predicted, target):
  return (-torch.sum(logsoftmax(predicted) * target)) / predicted.shape[0] # divided by batch size

def train_network(model, config, game_storage, savefile):
  optimizer = optim.SGD(model.parameters(), lr = config.learning_rate, momentum = config.momentum, weight_decay = config.weight_decay)
  criterion1 = nn.MSELoss().to(DEVICE)
  criterion2 = softmax_crossentropy_with_logits
  model.train()

  for epoch in range(1, config.epochs_between_games + 1):
    data_batch = game_storage.get_batch()

    dl_params = {'batch_size' : len(data_batch),
                 'shuffle'    : False,
                 'num_workers': 0}
    training_generator = DataLoader(data_batch, **dl_params)
    running_loss = 0.0
    start = time.time()
    for local_batch, (target_values, target_policies) in training_generator:
      # putting stuff on cuda, and to half precision, if possible
      local_batch = local_batch.to(DEVICE)
      target_values = target_values.to(DEVICE)
      target_policies = target_policies.to(DEVICE)

      optimizer.zero_grad()
      values, policies = model(local_batch)

      loss1 = criterion1(values, target_values)
      loss2 = criterion2(policies, target_policies)
      loss = loss1 + loss2
      loss.backward()

      optimizer.step()

      savefile.combined_loss.append(loss.data.item())
      savefile.value_loss.append(loss1.data.item())
      savefile.policy_loss.append(loss2.data.item())
      running_loss += loss.data.item()

    print(f'{epoch + savefile.num_epochs_trained}th epoch (batch loss: {running_loss:0.6f}, time: {(time.time() - start):0.4f} sec)')
  
  model.eval()

def pp_node(node, config):
  a = f'a: {node.action:02d}, '
  val = f'avg_val: {node.value:.6f}, '
  ucb_score = f'ucb: {ucb(node):.4f}, '
  prior = f'p: {node.prior:.4f}, '
  noise = f'noise: {node.noise:.4f}, '
  visits = f'{node.n}/{node.parent.n}'
  print(a + val + ucb_score + prior + noise + visits)

def pp_setup(savefile):
  config = savefile.config
  print('╔═══════════════════════════════════════════════════════════════════════════════╗') # len = 81
  print('║' + (' ' * 17) +'Neural Network (Assisted) Sequence Calculator' + (' ' * 17) + '║')
  print('╠═══════════════════════════════════════════════════════════════════════════════╣')
  s = (f'║ version: {savefile.version}')
  print(s + ' ' * (80 - len(s)) + '║')
  s = (f'║ games played: {savefile.num_games_played}')
  print(s + ' ' * (80 - len(s)) + '║')
  s = (f'║ epochs trained: {savefile.num_epochs_trained}')
  print(s + ' ' * (80 - len(s)) + '║')  
  s = (f'║ device: {DEVICE}')
  print(s + ' ' * (80 - len(s)) + '║')
  s = (f'║ num simul: {config.num_iterations}')
  print(s + ' ' * (80 - len(s)) + '║')
  s = (f'║ res blocks: {config.num_res_blocks}')
  print(s + ' ' * (80 - len(s)) + '║')
  s = (f'║ conv filters: {config.num_conv_filters}')
  print(s + ' ' * (80 - len(s)) + '║')
  s = (f'║ action size: {config.size}x{config.size} = {config.action_size}')
  print(s + ' ' * (80 - len(s)) + '║')
  s = (f'║ batch size: {config.batch_size}')
  print(s + ' ' * (80 - len(s)) + '║')  
  s = (f'║ window size: {config.window_size}')
  print(s + ' ' * (80 - len(s)) + '║')
  s = (f'║ learning rate: {config.learning_rate}')
  print(s + ' ' * (80 - len(s)) + '║')
  s = (f'║ games between training: {config.num_actors}')
  print(s + ' ' * (80 - len(s)) + '║')
  s = (f'║ epochs between games: {config.epochs_between_games}')
  print(s + ' ' * (80 - len(s)) + '║')
  s = (f'║ dirichlet alpha explor: {config.root_dirichlet_alpha}')
  print(s + ' ' * (80 - len(s)) + '║')
  s = (f'║ softmax sampling moves: {config.num_softmax_sampling_moves}')
  print(s + ' ' * (80 - len(s)) + '║')
  s = (f'║ use utility: {config.use_utility}')
  print(s + ' ' * (80 - len(s)) + '║')
  s = (f'║ use q value: {config.use_q_value}')
  print(s + ' ' * (80 - len(s)) + '║')
  s = (f'║ use full pre-activation: {config.use_full_pre_activation}')
  print(s + ' ' * (80 - len(s)) + '║')
  print('╚═══════════════════════════════════════════════════════════════════════════════╝')

def draw_plots(save):
  plt.clf()
  plt.figure(1)
  plt.subplot(221)
  plt.plot(save.combined_loss)
  plt.title('Combined loss')
  plt.grid(True)

  plt.subplot(222)
  plt.plot(save.value_loss)
  plt.title('Value loss')
  plt.grid(True)

  plt.subplot(223)
  plt.plot(save.policy_loss)
  plt.title('Policy loss')
  plt.grid(True)
  plt.draw()
  plt.pause(1) 