from __future__ import print_function
import os
import chess
# import torch
import time
import chess.svg
import traceback
import base64
from state import State

# Maximum assigned value to the evaluation function
MAXVAL = 10000



# Class to evaluate the board position
class Valuator(object):

  # Constructor loads the model
  def __init__(self):
    import torch
    from train import Net
    self.reset()
    vals = torch.load("nets/value.pth", map_location=lambda storage, loc: storage)
    self.model = Net()
    self.model.load_state_dict(vals)

  # Function called when the object is called
  # Passes the board into the model and returns the value
  def __call__(self, s):
    self.reset()
    brd = s.serialize()[None]
    output = self.model(torch.tensor(brd).float())
    return float(output.data[0][0])
  # count; to count the number of nodes visited
  # Function to reset count number
  def reset(self):
    self.count = 0



# Class to evaluate the board position
class ClassicValuator(object):

  # Assign seperate values to each piece
  values = {chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0}

  # Constructor resets the evaluation function and initializes memo
  def __init__(self):
    self.reset()
    self.memo = {}

  # Function to reset count number
  def reset(self):
    self.count = 0

  # Function called when the object is called
  # Passes the board into the model and returns the value
  def __call__(self, s):
    self.count += 1
    key = s.key()
    # Checking if node is in memory
    if key not in self.memo:
      self.memo[key] = self.value(s)
    return self.memo[key]

  # Function to return value of node not in memory
  def value(self, s):
    b = s.board

    # Game over values
    if b.is_game_over():
      if b.result() == "1-0":
        return MAXVAL
      elif b.result() == "0-1":
        return -MAXVAL
      else:
        return 0
    
    # Initialize value
    val = 0.0
    # Get a dictionary of pieces
    pm = s.board.piece_map()

    # Iterate over each piece
    for x in pm:
      # Assign value for each piece
      tval = self.values[pm[x].piece_type]
      # If it a white piece add to value
      if pm[x].color == chess.WHITE:
        val += tval
      # If it a black piece detract from value
      else:
        val -= tval

    # Add value for each legal move: white
    bak = b.turn
    b.turn = chess.WHITE
    val += 0.1 * b.legal_moves.count()
    # Detract value for each legal move: black
    b.turn = chess.BLACK
    val -= 0.1 * b.legal_moves.count()
    b.turn = bak

    return val



# Mini-max algorithm
def computer_minimax(s, v, depth, a, b, big=False):

  # If depth > 5 => too costly to venture further nodes
  # or game over return value
  if depth >= 5 or s.board.is_game_over():
    return v(s)

  # Set return value based on who's turn it is
  turn = s.board.turn
  if turn == chess.WHITE:
    ret = -MAXVAL
  else:
    ret = MAXVAL
  if big:
    bret = []

  # Add all the neighbouring nodes to a list
  isort = []
  for e in s.board.legal_moves:
    s.board.push(e)
    isort.append((v(s), e))
    s.board.pop()
  move = sorted(isort, key=lambda x: x[0], reverse=s.board.turn)

  # If depth is beyond 3 then only view the most promising nodes
  # as it is too costly to view all nodes
  if depth >= 3:
    move = move[:10]

  # Iterate over each legal move
  for e in [x[1] for x in move]:
    
    # Perform the move on the board 
    s.board.push(e)

    #  Recursive call to the mini-max algorithm to get the value of the 
    # neighbouring nodes
    tval = computer_minimax(s, v, depth+1, a, b)
    
    # Remove the move 
    s.board.pop()
    if big:
      bret.append((tval, e))

    # Alpha-beta pruning
    if turn == chess.WHITE:
      ret = max(ret, tval)
      a = max(a, ret)
      if a >= b:
        break  # Prune alpha
    else:
      ret = min(ret, tval)
      b = min(b, ret)
      if a >= b:
        break  # Prune beta

  # Return the value
  if big:
    return ret, bret
  else:
    return ret

# Function to explore the nodes and return the best move
def explore_leaves(s, v):
  ret = []
  start = time.time()
  v.reset()
  bval = v(s)
  cval, ret = computer_minimax(s, v, 0, a=-MAXVAL, b=MAXVAL, big=True)
  eta = time.time() - start
  print("%.2f -> %.2f: explored %d nodes in %.3f seconds %d/sec" % (bval, cval, v.count, eta, int(v.count/eta)))
  return ret

# chess board and "engine"
s = State()
# v = Valuator()
v = ClassicValuator()



# Convert chess board into SVG graphical format 
def to_svg(s):
  return base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')

from flask import Flask, Response, request
app = Flask(__name__)

@app.route("/")
def hello():
  ret = open("index.html").read()
  return ret.replace('start', s.board.fen())


# Function to make the computer move a piece
def computer_move(s, v):
  move = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)
  if len(move) == 0:
    return
  print("top 3:")
  for _,m in enumerate(move[0:3]):
    print("  ",m)
  print(s.board.turn, "moving", move[0][1])
  s.board.push(move[0][1])



# move given in algebraic notation
@app.route("/move")
def move():
  if not s.board.is_game_over():
    move = request.args.get('move',default="")
    if move is not None and move != "":
      print("human moves", move)
      try:
        s.board.push_san(move)
        computer_move(s, v)
      except Exception:
        traceback.print_exc()
      response = app.response_class(
        response=s.board.fen(),
        status=200
      )
      return response
  else:
    print("GAME IS OVER")
    response = app.response_class(
      response="game over",
      status=200
    )
    return response
  print("hello ran")
  return hello()

# moves given as coordinates of piece moved
@app.route("/move_coordinates")
def move_coordinates():
  if not s.board.is_game_over():
    source = int(request.args.get('from', default=''))
    target = int(request.args.get('to', default=''))
    promotion = True if request.args.get('promotion', default='') == 'true' else False

    move = s.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

    if move is not None and move != "":
      print("human moves", move)
      try:
        s.board.push_san(move)
        computer_move(s, v)
      except Exception:
        traceback.print_exc()
    response = app.response_class(
      response=s.board.fen(),
      status=200
    )
    return response

  print("GAME IS OVER")
  response = app.response_class(
    response="game over",
    status=200
  )
  return response

@app.route("/newgame")
def newgame():
  s.board.reset()
  response = app.response_class(
    response=s.board.fen(),
    status=200
  )
  return response


if __name__ == "__main__":
  app.run(debug=True)


