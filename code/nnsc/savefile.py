import pickle, math, datetime, os, torch
import nnsc.gamestorage as gamestorage
import nnsc.network as net

PATH = 'nnsc/saves/'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################
####### Savefile class #########
################################
class Savefile():
  def __init__(self, config,
                     network = None,
                     game_storage = None, 
                     num_epochs_trained = 0, 
                     num_games_played = 0, 
                     time_spent = 0, 
                     combined_loss = [], 
                     value_loss = [], 
                     policy_loss = [],
                     save_mode = False,
                     version = ''):
    self.config = config
    self.network = network if network else net.Network(config)
    self.network = self.network.to(DEVICE) # not sure if this should be done here

    self.game_storage = game_storage if game_storage else gamestorage.GameStorage(config)
    self.num_epochs_trained = num_epochs_trained
    self.num_games_played = num_games_played
    self.time_spent = time_spent
    self.combined_loss = combined_loss
    self.value_loss = value_loss
    self.policy_loss = policy_loss
    self.save_mode = save_mode
    self.version = version

  def update(self, time_spent):
    self.time_spent += time_spent
    self.num_games_played += self.config.num_actors
    self.num_epochs_trained += self.config.epochs_between_games

def save_to_disk(savefile):
  t = datetime.datetime.now()
  filename = f'{t.year}{t.month:02d}{t.day:02d}-{t.hour:02d}{t.minute:02d}-{t.second:02d}{t.microsecond:06d}-epoch{savefile.num_epochs_trained}'
  path = PATH + savefile.version + '/'
  # put network on CPU, so that it can also be opened and read on computers without a GPU
  with open(path + filename + '.save', 'wb') as file:
    pickle.dump(savefile, file)
  with open(path + 'info.txt', 'w') as info:
    info.writelines(get_info_lines(savefile))
  
def make_folder(savefile):
  path = PATH + savefile.version + '/'
  if not os.path.exists(path):
    try: os.mkdir(path)
    except OSError: print(f'Failed creating folder {path}')
    else: print(f'Successfully created folder {path}')

def load_from_disk(version):
  path = PATH + version + '/'
  files = [f for f in os.listdir(path) if '.save' in f]
  files.sort(reverse = True)
  if len(files) < 1:
    print(f'nothing to load from directory {version}/')
    exit(0)
  with open(path + files[0], 'rb') as file:
    return pickle.load(file)


def get_info_lines(savefile):
  path = PATH + savefile.version + '/'
  txt = [f'date: {datetime.datetime.now()}']
  txt.append(f'\nsimulations: {savefile.config.num_iterations}')
  txt.append(f'\nboard_size: {savefile.config.size}x{savefile.config.size}')
  txt.append(f'\nconvolutional filters: {savefile.config.num_conv_filters}')
  txt.append(f'\nresidual blocks: {savefile.config.num_res_blocks}')
  txt.append(f'\nlearning rate: {savefile.config.learning_rate}')
  txt.append(f'\ngames between training: {savefile.config.num_actors}')
  txt.append(f'\nepochs between games: {savefile.config.epochs_between_games}')
  txt.append(f'\ngame window size: {savefile.config.window_size}')
  txt.append(f'\nbatch size: {savefile.config.batch_size}')

  txt.append(f'\n\ntime elapsed: {datetime.timedelta(seconds = savefile.time_spent)} ({savefile.time_spent:.0f} seconds)')
  txt.append(f'\ngames played: {savefile.num_games_played}')
  txt.append(f'\nepochs trained: {savefile.num_epochs_trained}')
    
  if(len(savefile.combined_loss) > 0): txt.append(f'\n\nlatest combined loss: {savefile.combined_loss[-1]}')
  if(len(savefile.value_loss) > 0): txt.append(f'\nlatest value loss: {savefile.value_loss[-1]}')
  if(len(savefile.policy_loss) > 0): txt.append(f'\nlatest policy loss: {savefile.policy_loss[-1]}')
  return txt
