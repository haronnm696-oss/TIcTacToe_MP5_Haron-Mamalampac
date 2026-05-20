"""
================================================================================
  Tic-Tac-Toe with Minimax Algorithm
  Machine Problem #5
================================================================================
  The AI uses the Minimax algorithm to play optimally.
  It is impossible to beat the AI — the best a human player can achieve is a Draw.
================================================================================
"""

import math  # We use math.inf for representing +infinity and -infinity


# ──────────────────────────────────────────────────────────────────────────────
#  BOARD REPRESENTATION
# ──────────────────────────────────────────────────────────────────────────────
# The board is a list of 9 cells (index 0–8), laid out like this:
#
#   0 | 1 | 2
#   ---------
#   3 | 4 | 5
#   ---------
#   6 | 7 | 8
#
# Each cell holds ' ' (empty), 'X', or 'O'.
# The AI always plays as 'O'; the human plays as 'X'.

AI_PLAYER  = 'O'   # Maximizer — AI tries to MAXIMIZE its score
HUMAN_PLAYER = 'X' # Minimizer — Human is assumed to MINIMIZE AI's score

# All 8 possible winning combinations (rows, columns, diagonals)
WIN_COMBOS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
    [0, 4, 8], [2, 4, 6]               # diagonals
]


# ──────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def create_board():
    """Return a fresh, empty 9-cell board."""
    return [' '] * 9


def print_board(board):
    """Render the board in a human-friendly grid format."""
    print()
    for row in range(3):
        cells = board[row * 3 : row * 3 + 3]
        print(f"  {cells[0]} | {cells[1]} | {cells[2]} ")
        if row < 2:
            print("  ---------")
    print()


def print_board_with_indices():
    """Show the position indices so the human knows which number to enter."""
    print("\n  Position reference:")
    print("  0 | 1 | 2")
    print("  ---------")
    print("  3 | 4 | 5")
    print("  ---------")
    print("  6 | 7 | 8")
    print()


def check_winner(board, player):
    """
    Return True if `player` has three in a row on the board.
    We check all 8 winning combinations.
    """
    for combo in WIN_COMBOS:
        if all(board[i] == player for i in combo):
            return True
    return False


def get_empty_cells(board):
    """Return a list of indices where the board cell is still empty (' ')."""
    return [i for i, cell in enumerate(board) if cell == ' ']


def is_board_full(board):
    """Return True if there are no empty cells left (potential draw)."""
    return len(get_empty_cells(board)) == 0


def is_terminal(board):
    """
    Return True when the game is over — either someone has won,
    or the board is completely filled (draw).
    """
    return check_winner(board, AI_PLAYER) or \
           check_winner(board, HUMAN_PLAYER) or \
           is_board_full(board)


def evaluate(board):
    """
    Static evaluation function — assigns a numeric score to a board state.

    Scoring convention (from the AI's point of view):
        +10  →  AI ('O') wins         — great outcome for AI
         -10  →  Human ('X') wins      — bad outcome for AI
          0   →  Draw or game ongoing  — neutral

    The Minimax algorithm calls this on terminal (leaf) nodes of the game tree.
    """
    if check_winner(board, AI_PLAYER):
        return 10
    if check_winner(board, HUMAN_PLAYER):
        return -10
    return 0  # Draw or non-terminal (shouldn't be called on non-terminal)


# ──────────────────────────────────────────────────────────────────────────────
#  MINIMAX ALGORITHM (with Alpha-Beta Pruning)
# ──────────────────────────────────────────────────────────────────────────────

def minimax(board, depth, is_maximizing, alpha, beta):
    """
    The core Minimax algorithm with Alpha-Beta Pruning.

    How it works:
    ─────────────
    Minimax simulates ALL possible future moves from the current board state,
    building a full game tree down to terminal nodes (win/draw/loss).

    • When it is the AI's turn  (is_maximizing = True):
        The AI picks the move that leads to the HIGHEST score.
    • When it is the human's turn (is_maximizing = False):
        The AI assumes the human plays perfectly and picks the move
        that leads to the LOWEST score (worst for the AI).

    By recursively alternating these two perspectives, the AI always finds
    the optimal strategy.

    Alpha-Beta Pruning:
    ───────────────────
    A performance optimization. We track two bounds:
      • alpha — the best score the MAXIMIZER (AI) is guaranteed so far.
      • beta  — the best score the MINIMIZER (Human) is guaranteed so far.
    If at any node beta ≤ alpha, we PRUNE (skip) the remaining branches
    because they cannot possibly affect the final decision. This dramatically
    reduces the number of nodes evaluated without changing the result.

    Parameters:
        board          : current board state (list of 9 cells)
        depth          : how many levels deep we are in the recursion tree
                         (used to prefer faster wins — higher score for quicker wins)
        is_maximizing  : True if it's the AI's turn, False if it's the human's
        alpha          : best score found so far for the maximizer (starts at -∞)
        beta           : best score found so far for the minimizer (starts at +∞)

    Returns:
        The best achievable score (integer) from this board state.
    """

    # ── Base case: if the game is over, return the board's score ──────────────
    if is_terminal(board):
        score = evaluate(board)
        # We subtract/add depth so the AI prefers QUICKER wins and LONGER losses.
        # e.g., winning in 1 move (+10 - 0 depth) is better than winning in 3 (+10 - 2 depth).
        if score == 10:
            return score - depth
        if score == -10:
            return score + depth
        return 0  # Draw

    empty_cells = get_empty_cells(board)

    # ── Maximizing turn: AI's perspective — choose the HIGHEST score ──────────
    if is_maximizing:
        best_score = -math.inf  # Start with the worst possible score for maximizer

        for cell in empty_cells:
            board[cell] = AI_PLAYER          # Try placing AI's mark
            score = minimax(board, depth + 1, False, alpha, beta)  # Recurse as minimizer
            board[cell] = ' '                # Undo the move (backtracking)

            best_score = max(best_score, score)
            alpha = max(alpha, best_score)   # Update alpha (maximizer's best)

            if beta <= alpha:
                break  # ✂️ Beta cutoff — prune remaining branches

        return best_score

    # ── Minimizing turn: Human's perspective — choose the LOWEST score ────────
    else:
        best_score = math.inf  # Start with the worst possible score for minimizer

        for cell in empty_cells:
            board[cell] = HUMAN_PLAYER       # Try placing human's mark
            score = minimax(board, depth + 1, True, alpha, beta)   # Recurse as maximizer
            board[cell] = ' '               # Undo the move (backtracking)

            best_score = min(best_score, score)
            beta = min(beta, best_score)    # Update beta (minimizer's best)

            if beta <= alpha:
                break  # ✂️ Alpha cutoff — prune remaining branches

        return best_score


def find_best_move(board):
    """
    Determine the best possible move for the AI.

    We iterate over every empty cell and run minimax() for each one,
    then pick the cell that produces the highest score.

    Returns:
        best_cell (int): the index (0–8) of the AI's optimal move.
    """
    best_score = -math.inf
    best_cell  = -1

    for cell in get_empty_cells(board):
        board[cell] = AI_PLAYER   # Temporarily place AI's mark
        # Call minimax starting at depth=0, next turn is the human (minimizing)
        score = minimax(board, 0, False, -math.inf, math.inf)
        board[cell] = ' '        # Undo the move

        if score > best_score:
            best_score = score
            best_cell  = cell

    return best_cell


# ──────────────────────────────────────────────────────────────────────────────
#  GAME LOOP
# ──────────────────────────────────────────────────────────────────────────────

def get_human_move(board):
    """
    Prompt the human player to enter a valid move (0–8).
    Keeps asking until a valid, unoccupied cell is chosen.
    """
    while True:
        try:
            move = int(input("  Your move (0-8): "))
            if move < 0 or move > 8:
                print("  ⚠  Please enter a number between 0 and 8.")
            elif board[move] != ' ':
                print("  ⚠  That cell is already taken. Choose another.")
            else:
                return move
        except ValueError:
            print("  ⚠  Invalid input. Please enter a number (0–8).")


def play_game():
    """
    Main game loop. Handles turn order, win/draw detection, and display.
    """
    print("=" * 50)
    print("     TIC-TAC-TOE  —  Minimax AI")
    print("=" * 50)
    print("  You are X.  The AI is O.")
    print("  Try to beat the AI... (good luck!)")
    print_board_with_indices()

    board = create_board()

    # Decide who goes first
    while True:
        choice = input("  Do you want to go first? (y/n): ").strip().lower()
        if choice in ('y', 'n'):
            human_first = (choice == 'y')
            break
        print("  Please enter 'y' or 'n'.")

    # Determine the turn order
    # turn = 'human' or 'ai'
    turn = 'human' if human_first else 'ai'

    print()
    print_board(board)

    # ── Main game loop ─────────────────────────────────────────────────────────
    while not is_terminal(board):

        if turn == 'human':
            print("  Your turn (X):")
            move = get_human_move(board)
            board[move] = HUMAN_PLAYER
            turn = 'ai'

        else:
            print("  AI is thinking...")
            move = find_best_move(board)
            board[move] = AI_PLAYER
            print(f"  AI placed O at position {move}.")
            turn = 'human'

        print_board(board)

    # ── Determine and announce the result ─────────────────────────────────────
    print("=" * 50)
    if check_winner(board, HUMAN_PLAYER):
        print("  🎉  You WIN! (This should never happen against a perfect AI!)")
    elif check_winner(board, AI_PLAYER):
        print("  🤖  AI WINS! Better luck next time.")
    else:
        print("  🤝  It's a DRAW! Well played.")
    print("=" * 50)


def main():
    """Entry point. Allows replaying without restarting the program."""
    while True:
        play_game()
        again = input("\n  Play again? (y/n): ").strip().lower()
        if again != 'y':
            print("\n  Thanks for playing! Goodbye.\n")
            break
        print()


# ── Run the program ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()