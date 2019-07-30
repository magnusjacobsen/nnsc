import hex.gui as gui
import hex.game as game
import tkinter as tk
from nnsc import nnsc
import time, pickle, datetime

PATH = 'nnsc/saves/games/'

class Controller():
  def __init__(self, agent1, agent2, size, visual = False):
    self.gui = gui.View(size)
    self.agent1 = agent1
    self.agent2 = agent2
    self.size = size
    self.human_action = self.gui.human_action
    self.gui.set_controller(self)
    self.visual = visual

  def run(self, games_to_play):
    self._run(games_to_play)
    if self.visual: self.gui.root.mainloop()

  def _run(self, games_to_play):
    num_as_snd = games_to_play // 2
    num_as_fst = games_to_play - num_as_snd
    start_time = time.time()
    wins_as_fst = self.play_games(self.agent1, self.agent2, num_as_fst)
    wins_as_snd = (num_as_snd - self.play_games(self.agent2, self.agent1, num_as_snd, games_played = num_as_fst))
    self.pp_winrates(wins_as_fst, wins_as_snd, games_to_play, time.time() - start_time)

  # for this to work, agent1 needs to be a list of agents
  def run_many(self, agent_filelist, opponent, games_to_play):
    self.agent2 = opponent
    for filename in agent_filelist:
      self.agent1 = get_nn_agent(filename)
      self._run(games_to_play)
    if self.visual: self.gui.root.mainloop()

  def human_input(self):
    self.gui.unfreeze()
    self.gui.root.wait_variable(self.human_action)
    self.gui.freeze()
    return [self.human_action.get()]

  def play_games(self, fst, snd, num_games, games_played = 0):
    states = [game.initial_state(self.size) for _ in range(num_games)]
    fst_wins = 0
    turn = 0
    finished_games = games_played
    while states:
      if self.visual: self.gui.update(states[0])
      if turn % 2 == 0: # it's first players' turn
        if fst: # fst is an AI
          actions = fst.decide_parallel(states)
        else:
          actions = self.human_input()
      else: 
        if snd: # snd is an AI
          actions = snd.decide_parallel(states)
        else:
          actions = self.human_input()

      states = [game.result(s, a) for s, a in zip(states, actions)]
      if self.visual: self.gui.update(states[0])
    
      fst_wins += sum(1 for s in states if game.utility(s, 1) == 1)
      new_states = []
      for s in states:
        if game.terminal_test(s):
          finished_games += 1
          #self.save_game_result(fst, snd, game.utility(s,1), finished_games)
        else:
          new_states.append(s)
      states = new_states # important step ;)
      turn += 1
    return fst_wins

  def save_game_result(self, fst, snd, result, game_number):
    print('saving game result')
    fst = f'{fst.version}_{fst.epoch}' if fst else 'Human'
    snd = f'{snd.version }_{snd.epoch}' if snd else 'Human'
    res = '1-0' if result == 1 else '0-1'
    filename = PATH + fst + '-' + snd + '_game_num_' +  str(game_number) + '.pgn'
    txt = [f'[Date "{datetime.datetime.now()}"]']
    txt.append(f'\n[White "{fst}"]')
    txt.append(f'\n[Black "{snd}"]')
    txt.append(f'\n[Result "{res}"]')
    txt.append('\n.\n\n')
    with open(filename, 'w') as info:
      info.writelines(txt)

  def pp_winrates(self, wins_as_fst, wins_as_snd, games_played, time_elapsed):
    num_as_snd = games_played // 2
    num_as_fst = games_played - num_as_snd
    wins = wins_as_fst + wins_as_snd
    losses_as_fst = num_as_fst - wins_as_fst
    losses_as_snd = num_as_snd - wins_as_snd
    opp_wins = games_played - wins
    winrate = wins / games_played
    print('╔═══════════════════════════════════════════════════════════════════════════════╗') # len = 81
    player = str(self.agent1) if self.agent1 else 'Human'
    s = f'║ {player}'
    print(s + ' ' * (80 - len(s)) + '║')
    s = f'║ winrate: {winrate} ({wins}/{games_played}, fst: {wins_as_fst}/{num_as_fst}, snd: {wins_as_snd}/{num_as_snd})'
    print(s + ' ' * (80 - len(s)) + '║')
    print('╠═══════════════════════════════════════════════════════════════════════════════╣')
    opp_winrate = opp_wins / games_played
    player = str(self.agent2) if self.agent2 else 'Human'
    s = f'║ {player}'
    print(s + ' ' * (80 - len(s)) + '║')
    s = f'║ winrate: {opp_winrate} ({opp_wins}/{games_played}, fst: {losses_as_snd}/{num_as_snd}, snd: {losses_as_fst}/{num_as_fst})'
    print(s + ' ' * (80 - len(s)) + '║')
    print('╠═══════════════════════════════════════════════════════════════════════════════╣')
    finished = time.time()
    s = f'║ time: {time_elapsed:.4f} ({time_elapsed / games_played:.4f} secs per game)'
    print(s + ' ' * (80 - len(s)) + '║')
    print('╚═══════════════════════════════════════════════════════════════════════════════╝')

def get_nn_agent(path):
  version = path.split('saves/')[1].split('/')[0]
  with open(path, 'rb') as f:
    savefile = pickle.load(f)
    print('**********************')
    if not hasattr(savefile.config, 'use_half_precision'):
      savefile.config.use_half_precision = True
    print(path)
  return nnsc.AI(savefile)