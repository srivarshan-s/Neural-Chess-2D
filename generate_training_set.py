import os
import chess.pgn
import numpy as np
from state import State


# Function to load training data by parsing a given PGN file
def get_dataset():

  # X, Y: Training data for the model
  X,Y = [], []
  # Game number
  gn = 0
  # Values to the results of the game i.e. draw, loss or win
  values = {'1/2-1/2':0, '0-1':-1, '1-0':1}

  # Read the .pgn files in the data/ directory
  # A .pgn file is a text based file format to record a chess match
  for fn in os.listdir("data"):
    # Read the PGN file
    pgn = open(os.path.join("data", fn))

    while 1:
      # Load the chess game from the PGN file  
      game = chess.pgn.read_game(pgn)
      # Break if the game is corrupted
      if game is None:
        break
      # Get the result of the game 
      res = game.headers['Result']
      # If result of game is not draw, loss or win
      # we continue to the next game
      if res not in values:
        continue
      # Get the result value of the game i.e. 0, -1 or 1 from the
      # value dictionary
      value = values[res]
      # Get the initial chess board configuration
      # i.e. position of all the pieces in the board
      board = game.board()

      # Iterate over every single move made in the game
      for _, move in enumerate(game.mainline_moves()):
        # Change the board according to the move
        board.push(move)
        # Serialize the game/board state
        ser = State(board).serialize()
        # Insert the serialized board and the result of the match
        # into the training data
        X.append(ser)
        Y.append(value)

      print("parsing game %d, got %d examples" % (gn, len(X)))

      # Increment the game number as we move on to the next game
      gn += 1

  # Convert the training data into a numpy array
  X = np.array(X)
  Y = np.array(Y)

  return X,Y



if __name__ == "__main__":

  # Load the training data
  X,Y = get_dataset() 
  # Save the training data as a NPZ file
  np.savez("processed/dataset.npz", X, Y) 

