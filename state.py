import chess
import numpy as np



class State(object):

  # Constructor; initializes the object
  def __init__(self, board=None):

    # If no chess board is passed, then the object loads a new chess board
    if board is None:
      self.board = chess.Board()
    # The board passed as arguement is loaded
    else:
      self.board = board

  # Function to get some important details of the cheass board
  def key(self):
    # board_fen(); returns board position in the FEN notation
    # turn; return the side to mode (white or black)
    # castling_rights; returns bitmask of rooks with ability to castle
    # ep_square; returns a potential en passant square move
    return (self.board.board_fen(), self.board.turn, self.board.castling_rights, self.board.ep_square)

  def serialize(self):
    # Continues execution only if the board is valid
    assert self.board.is_valid()

    # Initializes the board state
    bstate = np.zeros(64, np.uint8)
    # Iterates over each square on the chess board
    for i in range(64):

      # Return the piece at that particular square
      pp = self.board.piece_at(i)

      if pp is not None:
        # Piece.symbol(); returns the symbol for the pieces where capital
        # leters denote white pieces and small leters denote black pieces
        bstate[i] = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, \
                     "p": 9, "n":10, "b":11, "r":12, "q":13, "k": 14}[pp.symbol()]

    # Special numbers for castling rights are assigned
    if self.board.has_queenside_castling_rights(chess.WHITE):
      assert bstate[0] == 4
      bstate[0] = 7
    if self.board.has_kingside_castling_rights(chess.WHITE):
      assert bstate[7] == 4
      bstate[7] = 7
    if self.board.has_queenside_castling_rights(chess.BLACK):
      assert bstate[56] == 8+4
      bstate[56] = 8+7
    if self.board.has_kingside_castling_rights(chess.BLACK):
      assert bstate[63] == 8+4
      bstate[63] = 8+7

    # Special numbers for en passant moves capable squares are assigned
    if self.board.ep_square is not None:
      assert bstate[self.board.ep_square] == 0
      bstate[self.board.ep_square] = 8

    # Board state is reshaped into a 8x8 square
    bstate = bstate.reshape(8,8)

    # Initializes a board state in binary format
    state = np.zeros((5,8,8), np.uint8)

    # The board state is converted into binary
    state[0] = (bstate>>3)&1
    state[1] = (bstate>>2)&1
    state[2] = (bstate>>1)&1
    state[3] = (bstate>>0)&1

    # The 4th column denotes whether it is black or whites turn
    state[4] = (self.board.turn*1.0)

    # The binary board state is returned
    return state

  # Function to provide a list of legal moves possible from current state
  def edges(self):
    # legal_moves; provides a dynamic list of legal moves
    return list(self.board.legal_moves)



if __name__ == "__main__":
  s = State()


