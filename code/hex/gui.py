import tkinter as tk
from PIL import Image, ImageTk
import math
import numpy as np
import hex.game as game

# constants
img_size = 40
margins = 20
block_size = img_size + margins
status_height = 60
status_width = 52

margin_x = 60
margin_y = 60
degrees = 45.
rad = math.radians(degrees)
y_scalar = 0.6

color_bg = '#f9eccc'

def rotate(x, y, center_x, center_y, rad):
  x, y = x - center_x + 30, y - center_y + 30
  dx = (x * math.cos(rad)) - (y * math.sin(rad))
  dy = (y * math.cos(rad)) + (x * math.sin(rad))
  x, y = dx + center_x - 20, dy + center_y - 20
  return x, y

def centroid(a, b, c):
  x = (a[0] + b[0] + c[0]) / 3
  y = (a[1] + b[1] + c[1]) / 3
  return x, y

def load_image(path):
  img = Image.open(path)
  img = ImageTk.PhotoImage(img)
  return img

def image_label(c, img):
  img_label = tk.Label(c, image = img, borderwidth = 0, highlightthickness = 0)
  img_label.image = img
  return img_label

class View():
  def __init__(self, size):

    self.size = size
    self.hover_list = []
    self.o_size = self.size * block_size
    self.width = self.size * block_size * 1.4142 # 0.33% added by rotating the square
    self.height = self.size * block_size * 1.4142
    self.center_x = self.width / 2
    self.center_y = self.height / 2
    self.canvas_height = self.height * y_scalar * 1.03 + (margin_y * 2)
    self.canvas_width = self.width + (margin_x * 2)

    self.root = tk.Tk()
    self.root.title('Hex ~~~ Con Tac Tix')
    self.root.deiconify()
    self.set_quit_key()

    self.canvas = tk.Canvas(self.root, width = self.canvas_width, height = self.canvas_height)
    self.canvas.configure(background = color_bg)
    self.canvas.pack()

    self.load_graphics()
    self.positions = self.draw_positions()
    self.create_status_indicator_lables()
    self.draw_status_indicators(self.create_centroids())
    self.game_state = None
    self.interactive_mode = False
    self.controller = None
    self.human_action = tk.IntVar()


  def set_quit_key(self):
    def key_pressed(event):
      if event.char == 'q':
        print('Quitter!')
        self.human_action.set(-1) # dirty hack, to get around wait_variable blocking the closing of the application
        self.root.destroy()
        exit(0)
    self.root.bind('<Key>', key_pressed)

  def set_controller(self, controller):
    self.controller = controller

  def create_centroids(self):
    self.maxx += 40
    self.maxy += 40
    self.minx -= 1
    self.miny -= 1
    a = self.minx, self.miny
    b = (self.maxx - self.minx) / 2, self.miny
    c = self.maxx, self.miny
    d = self.minx, (self.maxy - self.miny) / 2
    e = self.maxx, (self.maxy - self.miny) / 2
    f = self.minx, self.maxy
    g = (self.maxx - self.minx) / 2, self.maxy
    h = self.maxx, self.maxy
    x1, y1 = centroid(a, b, d)  # top-left triangle centroid
    x1, y1 = x1 - 40, y1 - 40
    x2, y2 = centroid(b, c, e)  # top-right
    x2, y2 = x2 + 40, y2 - 40
    x3, y3 = centroid(d, f, g)  # bottom-left
    x3, y3 = x3 - 40, y3 + 40
    x4, y4 = centroid(e, g, h)  # bottom-right
    x4, y4 = x4 + 40, y4 + 40
    return x1, y1, x2, y2, x3, y3, x4, y4

  def adjust_centroid_x(self, x):
    return x - (status_width / 2)

  def adjust_centroid_y(self, y):
    return y - (status_height / 2)

  def adjust_x(self, x):
    return x + margin_x

  def adjust_y(self, y):
    return y * y_scalar + self.o_size * 0.18 + margin_y

  def set_game_state(self, game_state):
    self.game_state = game_state

  def unfreeze(self):
    self.interactive_mode = True

  def freeze(self):
    self.interactive_mode = False

  def update(self, game_state):
    self.set_game_state(game_state)
    for i in range(len(self.positions)):
      owner = self.game_state.board[i]
      if owner == 1: 
        self.positions[i][3].lift()
      elif owner == -1: 
        self.positions[i][4].lift()
      else:
        self.positions[i][0].lift()

    self.status_one_opp_turn1.lift()
    self.status_one_opp_turn2.lift()
    self.status_two_opp_turn1.lift()
    self.status_two_opp_turn2.lift()
    if game.terminal_test(self.game_state):
      if game.player(self.game_state) == 1: #player -1 won
        self.status_two_won1.lift()
        self.status_two_won2.lift()
      else:
        self.status_one_won1.lift()
        self.status_one_won2.lift()
    else: # game is still running
      if game.player(self.game_state) == 1:
        self.status_one_turn1.lift()
        self.status_one_turn2.lift()
      else:
        self.status_two_turn1.lift()
        self.status_two_turn2.lift()
    self.canvas.update()

  def load_graphics(self):
    # board position graphics
    self.img_free = load_image('./hex/img/free.png')
    self.img_taken_one = load_image('./hex/img/taken_one.png')
    self.img_hover_one = load_image('./hex/img/hover_one.png')
    self.img_taken_two = load_image('./hex/img/taken_two.png')
    self.img_hover_two = load_image('./hex/img/hover_two.png')
    # status indicator
    self.img_status_one_turn = load_image('./hex/img/side_one_turn.png')
    self.img_status_one_opp_turn = load_image('./hex/img/side_one_opponent_turn.png')
    self.img_status_one_won = load_image('./hex/img/side_one_won.png')
    self.img_status_two_turn = load_image('./hex/img/side_two_turn.png')
    self.img_status_two_opp_turn = load_image('./hex/img/side_two_opponent_turn.png')
    self.img_status_two_won = load_image('./hex/img/side_two_won.png')

  def draw_positions(self):
    positions = []
    self.minx, self.maxx, self.miny, self.maxy = 100000, -1000000, 1000000, -1000000
    for i in range(self.size):
      for j in range(self.size):
        x = j * block_size
        y = i * block_size
        x, y = rotate(x, y, self.center_x, self.center_y, rad)
        y = self.adjust_y(y)
        x = self.adjust_x(x)

        index = i * self.size + j
        lst = self.create_many(x, y)
        self.bind_many(lst, index)
        self.place_many(lst, x, y)
        self.update_min_max_xy(x, y)
        positions.append(lst)
    return positions

  def update_min_max_xy(self, x, y):
    if self.maxx < x: self.maxx = x
    if self.minx > x: self.minx = x
    if self.maxy < y: self.maxy = y
    if self.miny > y: self.miny = y

  def create_many(self, x, y):
    lst = []
    lst.append(image_label(self.canvas, self.img_free))       # 0: free position
    lst.append(image_label(self.canvas, self.img_hover_one))  # 1: hovered by one
    lst.append(image_label(self.canvas, self.img_hover_two))  # 2: hovered by two
    lst.append(image_label(self.canvas, self.img_taken_one))  # 3: taken by one
    lst.append(image_label(self.canvas, self.img_taken_two))  # 4: taken by two        
    return lst

  def bind_many(self, lst, index):
    lst[0].bind('<Enter>', self.lambda_free_enter(lst[1], lst[2]))
    lst[1].bind('<Leave>', self.lambda_leave())
    lst[2].bind('<Leave>', self.lambda_leave())
    lst[3].bind('<Leave>', self.lambda_leave())
    lst[4].bind('<Leave>', self.lambda_leave())
    lst[1].bind('<Button-1>', self.lambda_click(index, lst[3], lst[4]))
    lst[2].bind('<Button-1>', self.lambda_click(index, lst[3], lst[4]))

  def place_many(self, lst, x, y):
    for elem in lst: elem.place(x = x, y = y)
    lst[0].lift() # make the free icon appear in front

  def create_status_indicator_lables(self):
    self.status_one_opp_turn1 = image_label(self.canvas, self.img_status_one_opp_turn)
    self.status_one_turn1 = image_label(self.canvas, self.img_status_one_turn)
    self.status_one_won1 = image_label(self.canvas, self.img_status_one_won)

    self.status_one_opp_turn2 = image_label(self.canvas, self.img_status_one_opp_turn)
    self.status_one_turn2 = image_label(self.canvas, self.img_status_one_turn)
    self.status_one_won2 = image_label(self.canvas, self.img_status_one_won)

    self.status_two_opp_turn1 = image_label(self.canvas, self.img_status_two_opp_turn)
    self.status_two_turn1 = image_label(self.canvas, self.img_status_two_turn)
    self.status_two_won1 = image_label(self.canvas, self.img_status_two_won)

    self.status_two_opp_turn2 = image_label(self.canvas, self.img_status_two_opp_turn)
    self.status_two_turn2 = image_label(self.canvas, self.img_status_two_turn)
    self.status_two_won2 = image_label(self.canvas, self.img_status_two_won)

  def draw_status_indicators(self, coords):
    x1, y1, x2, y2, x3, y3, x4, y4 = coords
    # player 1, top left
    self.status_one_turn1.place(x = x2, y = y2)
    self.status_one_won1.place(x = x2, y = y2)
    self.status_one_opp_turn1.place(x = x2, y = y2)
    self.status_one_opp_turn1.lift()
    # player 1, bottom right
    self.status_one_turn2.place(x = x3, y = y3)
    self.status_one_won2.place(x = x3, y = y3)
    self.status_one_opp_turn2.place(x = x3, y = y3)
    self.status_one_opp_turn2.lift()
    # player 2, bottom left
    self.status_two_turn1.place(x = x1, y = y1)
    self.status_two_won1.place(x = x1, y = y1)    
    self.status_two_opp_turn1.place(x = x1, y = y1)
    self.status_two_opp_turn1.lift()
    # player 2, top right
    self.status_two_turn2.place(x = x4, y = y4)
    self.status_two_won2.place(x = x4, y = y4)
    self.status_two_opp_turn2.place(x = x4, y = y4)
    self.status_two_opp_turn2.lift()

  def lower_all_hovered(self):
    for x in self.hover_list: x.lower()
    self.hover_list = []

  def on_click(self, index, fst, snd):
    if self.interactive_mode:
      self.human_action.set(index)
      #self.controller.set_action(index)

  def on_leave(self):
    self.lower_all_hovered()

  def on_enter(self, hov_one, hov_two):
    self.lower_all_hovered()
    if self.interactive_mode:
      if self.game_state:
        if not game.terminal_test(self.game_state):
          if game.player(self.game_state) == 1: 
            self.hover_list.append(hov_one)
            hov_one.lift()
          else: 
            self.hover_list.append(hov_two)
            hov_two.lift()

  def lambda_click(self, index, fst, snd):
    return lambda e: self.on_click(index, fst, snd)

  def lambda_free_enter(self, hover_one, hover_two):
    return lambda e: self.on_enter(hover_one, hover_two)

  def lambda_leave(self):
    return lambda e: self.lower_all_hovered()