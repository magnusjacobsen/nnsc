import math

################################
#### Configuration class #######
################################
class Config():
  def __init__(self, 
               size = 9, 
               num_simulations = 200, 
               num_res_blocks = 19, 
               num_conv_filters = 256, 
               num_actors = 50, 
               num_epochs = 10, 
               window_size = 1000,
               batch_size = 512,
               use_utility = False, # in evaluation function, when a state is a terminal state
               use_q_value = False, # as a training target for value
               use_full_pre_activation = False, # in neural network, type of layout for residual blocks
               use_half_precision = True):

    self.use_utility = use_utility
    self.use_q_value = use_q_value
    self.use_full_pre_activation = use_full_pre_activation
    self.use_half_precision = use_half_precision

    self.size = size # length of a side
    self.num_iterations = num_simulations# the number of iterations we do for each PUCT run
    max_game_length = size * size
    self.num_softmax_sampling_moves = math.ceil(max_game_length / 10)
    self.num_actors = num_actors
    self.max_runs = 10000
    self.runs_between_saves = 5
    self.epochs_between_games = num_epochs

    # network
    self.num_conv_filters = num_conv_filters
    self.conv_stride = 1
    self.action_size = self.size * self.size
    self.num_res_blocks = num_res_blocks

    # exploration
    # with 0.03, on 5x5 board, it was so small, that using the same network, the same sequence of actions were played.
    self.root_noise_exploration_fraction = 0.25 # not sure
    # dirichlet_alpha as a function of average legal moves
    avg_legal_moves = (size * size) * 0.7
    self.root_dirichlet_alpha = 10 / avg_legal_moves #math.exp(-0.04 * avg_legal_moves)
    self.exploration_constant = 1.27

    # Game Storage
    self.window_size = window_size # aka window_size
    self.batch_size = batch_size

    # Training
    self.momentum = 0.9 
    self.weight_decay = 0.0001 # 1e-4
    self.learning_rate = 0.05 # TODO be a function of epochs and avg legal moves
