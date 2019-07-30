import os, torch, datetime
import hex.game as game
import numpy as np
from torch.utils import data

#################################
##### Saved Game data class #####
#################################
class SavedGame():
  def __init__(self, state, root, config):
    self.use_q_value = config.use_q_value
    self.size = config.size
    self.history = state.history
    self.child_visits = np.array([x[1] for x in root.stats])
    self.child_visits = self.child_visits.astype(np.float16)
    if self.use_q_value: # for when config.use_q_value is true
      self.q_values = [x[0] for x in root.stats]
    else: # for when config.use_q_value is false
      self.utility = (game.utility(state, 1), game.utility(state, -1))

  def image_stack(self, step : int):
    return torch.from_numpy(game._image_stack(self.history, self.size, step))

  def get_target_policy(self, step : int):
    if step % 2 == 0: return torch.from_numpy(self.child_visits[step]).float()
    else: # transposing out of place
      temp = []
      for i in range(len(self.child_visits[step])):
        transp_index = (i % self.size) * self.size + (i // self.size)
        temp.append(self.child_visits[step][transp_index])
      return torch.tensor(temp).float()

  def get_target_value(self, step : int):
    if self.use_q_value:
      return torch.tensor([self.q_values[step]]).float() # it is automatically set to the perspective of the player who took an action to this state
    else:
      return torch.tensor([self.utility[step % 2]]).float() # utility[0] for player 1, and utility[1] for player 2 
      

################################
######### Game Batch ###########
################################
class Batch(data.Dataset):
  def __init__(self, game_buffer, batch_size, turn_sum):
    p = [len(g.history) / turn_sum for g in game_buffer] # pick probability
    games = np.random.choice(game_buffer, size = batch_size, p = p)
    game_steps = [(g, np.random.randint(len(g.history))) for g in games]
    self.batch_size = batch_size
    self.data, self.labels = [], []
    for g, step in game_steps:
      self.data.append(g.image_stack(step))
      self.labels.append((g.get_target_value(step), g.get_target_policy(step)))

  def __len__(self):
    return self.batch_size

  def __getitem__(self, index):
    return self.data[index], self.labels[index]

################################
######### GameStorage ##########
################################
class GameStorage():
  def __init__(self, config):
    self.config = config
    self.games_buffer = []
    self.turn_sum = 0
    self.num_games_played = 0

  def save_game(self, game : SavedGame):
    #if self.num_games_played < self.config.early_window_active:
    #  if len(self.games_buffer) >= self.config.early_window_size:
    #    removed = self.games_buffer.pop(0) # first in, first out
    #    self.turn_sum -= len(removed.history)
    #else:
    if len(self.games_buffer) >= self.config.window_size:
      removed = self.games_buffer.pop(0) # first in, first out (oldest gets removed)
      self.turn_sum -= len(removed.history)
    self.games_buffer.append(game)
    self.turn_sum += len(game.history)
    self.num_games_played += 1
    
  # get a random batch of images of games at random steps
  # Longer games have a higher probability of being chosen, because they have more steps
  def get_batch(self):
    return Batch(self.games_buffer, self.config.batch_size, self.turn_sum)
