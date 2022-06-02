from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from heapq import heappush, heappop
from math import inf
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import argparse
import json

"""
Nonograms Solvers
"""

Row = List[int]
State = List[Row]
Constraints = List[List[int]]

@dataclass
class Solver:
	"""
	Base class for our Solver, implementing some functions which we need
	for all (or most) implementations.
	""" 
	name: str
	height: int
	width: int
	rows: Constraints
	columns: Constraints
	nodes_generated: int = 0
	nodes_expanded: int = 0

	def init_state(self) -> State:
		""" Start with an empty board. All constraints valid. """
		return []

	def next_states(self, state: State) -> List[State]:
		""" Get next states, by completing the next row. """
		if len(state) == self.height:
			return []

		board = state
		idx_row = len(board)
		next_states = []
		for new_row in self.get_possible_colorings(self.rows[idx_row], self.width):
			new_board = deepcopy(board)
			new_board.append(new_row)
			next_states.append(new_board)

		# Possible culling of next states: (it is commented because it might be
		# more appropriate for an informed searched strategy, but the benefits
		# to DFS and BFS are discussed in README.pdf):
		# next_states = list(filter(self.check_columns, next_states))

		return next_states

	def is_solution(self, state: State) -> bool:
		""" Check if state is a solution. """
		return len(state) == self.height and self.check_columns(state)

	def get_possible_colorings(
		self, constraint: List[int], width: int
	) -> List[List[int]]:
		""" Get all possible colorings which satisfy given constraints, for a
		given length of row or column. """
		base = [0 for i in range(width)]
		pieces_length = constraint
		total = sum(pieces_length) + len(pieces_length) - 1
		possible_colorings = []

		def build_coloring(l: List[int], idx_piece: int, last_idx: int, used: int):
			""" Helper function, used to find all possible colorings. At given
			step, it places one piece on the line/column at all possible
			places and then through recursion, places the other pieces
			left. """
			if idx_piece == len(pieces_length):
				possible_colorings.append(l.copy())
				return

			length = pieces_length[idx_piece]
			if width - last_idx <= max(length + 1, total - used):
				# This piece doesn't fit or the other pieces on the right
				# won't fit (they need total-used indexes).
				return

			# Fill the first place possible for this piece.
			new_l = l.copy()
			for k in range(0, length):
				new_l[last_idx + 2 + k] = 1
			build_coloring(
				new_l, idx_piece + 1, last_idx + 1 + length, used + length + 1
			)

			# Shift the piece once to the right until end.
			for idx in range(last_idx + 2, width - length):
				new_l[idx] = 0
				new_l[idx + length] = 1
				build_coloring(new_l, idx_piece + 1, idx + length, used + length + 1)

		build_coloring(base, 0, -2, 0)
		return possible_colorings

	def check_columns(self, state: State) -> bool:
		""" Check column restrictions for given state. """
		board = state

		for j in range(len(board[0])):
			# Check column j:
			col = self.columns[j]  # column constraint
			k = 0  # index in col
			curr_piece = 0  # length of current piece in column

			for i in range(len(board)):
				if board[i][j] == 1:
					curr_piece += 1
					if k >= len(col) or curr_piece > col[k]:
						return False
				else:
					if curr_piece != 0 and curr_piece != col[k]:
						return False
					if curr_piece != 0:
						k += 1
						curr_piece = 0

			# Aditional check: make sure that we still have enough
			# rows to fill remaining spaces
			remaining = 0 if k >= len(col) else col[k] - curr_piece + sum(col[k + 1 :])
			if remaining > self.height - len(board):
				return False
		return True


	def solve_to_json(self) -> Dict[str, Any]:
		"""
		Solve board and send back statistics along with the solution.
		"""
		start_time = time()
		board = self.solve()
		end_time = time()

		return {
			"strategy": self.get_strategy_name(),
			"nodes generated": self.nodes_generated,
			"nodes expanded": self.nodes_expanded,
			"time": end_time - start_time,
			"solution": board
		}

"""
Uninformed Searches (BFS, DFS, Iterative Deepening)

Problem state: 2-dimensional array representing board's state (1=black, 0=white),
filled, until a certain row
Next states: all possible board's states after completing next row.
"""

class SolverBFS(Solver):
	""" Nonogram Solver which uses BFS strategy. """

	def solve(self) -> Optional[State]:
		q = deque()
		q.append(self.init_state())
		while q:
			state = q.popleft()
			next_states = self.next_states(state)

			self.nodes_expanded += 1
			self.nodes_generated += len(next_states)

			for next_state in next_states:
				if self.is_solution(next_state):
					return next_state
				q.append(next_state)
		return None

	def get_strategy_name(self) -> str:
		return "bfs"


class SolverDFS(Solver):
	""" Nonogram Solver which uses DFS strategy. """

	def solve(self, max_depth=inf) -> Union[Optional[State], int]:
		""" Here we return None if no solution possible, the solution
		board or a number representing max depth we reached. """
		stack = []
		stack.append((self.init_state(), 0))
		max_depth_reached = False
		while stack:
			state, depth = stack.pop()
			self.nodes_expanded += 1

			if self.is_solution(state):
				return state
			if depth > max_depth:
				max_depth_reached = True
				continue

			next_states = self.next_states(state)

			self.nodes_generated += len(next_states)

			for next_state in self.next_states(state):
				if self.is_solution(next_state):
					return next_state
				stack.append((next_state, depth + 1))
		return max_depth if max_depth_reached else None

	def get_strategy_name(self) -> str:
		return "dfs"


class SolverIterativeDeepening(SolverDFS):
	""" Nonogram Solver which uses the Iterative Deepening strategy.
	Notices that it inherits SolverDFS in order to call that
	strategy's solver for each level. """

	def solve(self) -> Optional[State]:
		max_depth = 0
		while max_depth < inf:
			ans = SolverDFS.solve(self, max_depth)
			if ans != max_depth:
				return ans
			max_depth += 1
		return None

	def get_strategy_name(self) -> str:
		return "iterative deepening"


"""
A* Algorithm

Problem state and transitions are defined the same as for uninformed
strategies.
"""

def continuity_heuristic(board: State) -> float:
	""" Check how much the last row added disrupts continuity"""
	if len(board) < 2:
		return 0

	disruptiveness = 0
	for j in range(len(board[0])):
		# check for column continuity:
		if board[-1][j] != board[-2][j]:
			# check for line continuity:
			if (
				(j > 0 and board[-1][j] != board[-1][j-1])
				and
				(j < len(board[0])-1 and board[-1][j] != board[-1][j+1])
			):
				disruptiveness += 0.5
			# check for diagonal continuity:
			if (
				(j > 0 and board[-1][j] != board[-2][j-1])
				and
				(j < len(board[0])-1 and board[-1][j] != board[-2][j+1])
			):
				disruptiveness += 0.5
	return disruptiveness


def balanced_heuristic(board: State) -> float:
	""" Check if last list is centered heavy or boundary heavy """
	if len(board) < 1:
		return 0

	width = len(board[0])

	# split last line in 3 parts and count number of filled elements in each
	left = sum(board[-1][:(width // 3)])
	center = sum(board[-1][(width // 3):(width // 3 * 2)])
	right = sum(board[-1][(width // 3 * 2):])

	if center == 0:
		# it is boundaries heavy
		return abs(left - right)
	else:
		# it might be center heavy or un-balanced so we return how unbalanced
		# it is
		return left + right


class SolverAStar(Solver):
	"""
	Nonogram Solver using A* Algorithm

	Choose one of the euristic functions below
	"""
	h: Callable[[State], float] = continuity_heuristic
	# h: Callable[[State], float] = balanced_heuristic

	def solve(self) -> Optional[State]:
		h = SolverAStar.h

		priority_queue = []
		heappush(priority_queue, (0, self.init_state()))

		while priority_queue:
			hi, state = heappop(priority_queue)

			if self.is_solution(state):
				return state

			next_states = [
				s for s in self.next_states(state)
				if self.check_columns(s)
			]

			self.nodes_expanded += 1
			self.nodes_generated += len(next_states)

			for next_state in next_states:
				hj = hi + h(next_state)
				heappush(priority_queue, (hj, next_state))

		return None

	def get_strategy_name(self) -> str:
		return "a*"

"""
MAC Algorithm

Variables: rows [Row0, Row1, ... Row_height-1] and columns
[Col0, Col1, ... Col_width-1]
Domains: possible colorings of a row, based on row constraints and possible
colorings of a column, based on column constraints.
All Arcs: (Row_i, Col_j) and (Col_j, Row_i) for every i in range(height) and
every j in range(width)

We'll populate our board one row at the time, starting from Row0.
For Row_i, We pick a value from its domain. For that value, we might have to
reduce the domains for columns (arc-reduce). We apply AC-3, queue being
initialized with arcs [(Row_i, Col_j) for j in range(0, width)]. If for a
column (let's say Col_j), its domain changes, add to queue arcs for all rows
to that column. And so on.
Thus, after picking a value for a row, we end up reducing domains for other
uninitialized rows.
"""

Vars = Dict[str, List[int]]
Domains = Dict[str, List[List[int]]]


class SolverCSP(Solver):
	""" Nonogram Solver using the Constraints Satisfaction strategy.
	It is implemented using a backtracking algorithm, that uses AC3
	at each step in order to mantain arc consistency. """

	def init_problem(self) -> Tuple[Vars, Domains]:
		# Initialize variables:
		var = {"rows": [], "cols": []}
		var["rows"] = list(range(0, self.height))
		var["cols"] = list(range(0, self.width))

		# Initialize domains:
		domains = {"rows": [], "cols": []}

		domains["rows"] = [[] for i in range(self.height)]
		for row in var["rows"]:
			domains["rows"][row] = self.get_possible_colorings(
				self.rows[row], self.width
			)
		domains["cols"] = [[] for j in range(self.width)]
		for col in var["cols"]:
			domains["cols"][col] = self.get_possible_colorings(
				self.columns[col], self.height
			)

		return (var, domains)

	def arc_reduce(self, var, domains, row: int, col: int) -> Tuple[bool, bool]:
		keep_val_row = [False for _ in range(len(domains["rows"][row]))]
		keep_val_col = [False for _ in range(len(domains["cols"][col]))]

		# Only keep rows and columns which can intersect in point [row][col] on board.
		for i, val_row in enumerate(domains["rows"][row]):
			for j, val_col in enumerate(domains["cols"][col]):
				if val_row[col] == val_col[row]:
					keep_val_row[i] = True
					keep_val_col[j] = True

		new_domain_row = [
			val for i, val in enumerate(domains["rows"][row]) if keep_val_row[i]
		]
		domain_row_changed = len(domains["rows"][row]) != len(new_domain_row)
		domains["rows"][row] = new_domain_row

		new_domain_col = [
			val for j, val in enumerate(domains["cols"][col]) if keep_val_col[j]
		]
		domain_col_changed = len(domains["cols"][col]) != len(new_domain_col)
		domains["cols"][col] = new_domain_col

		return domain_row_changed, domain_col_changed

	def arc_consistency(self, row, var, domains) -> bool:
		""" AC3, only starting with arcs from given row. """
		q = deque()
		# Add arcs from row to all columns:
		for col in var["cols"]:
			q.append((row, col))

		while q:
			row, col = q.popleft()

			domain_row_changed, domain_col_changed = self.arc_reduce(
				var, domains, row, col
			)
			if domain_row_changed:
				# Row's domain has changed. So we add all arcs from
				# that row.
				if len(domains["rows"][row]) == 0:
					return False
				q.extend(
					[
						(row, other_col)
						for other_col in range(self.width)
						if other_col != col
					]
				)
			if domain_col_changed:
				# Column's domain has changed. So we add all arcs from
				# that column.
				if len(domains["cols"][col]) == 0:
					return False
				q.extend(
					[
						(other_row, col)
						for other_row in range(self.height)
						if other_row != row
					]
				)
		return True

	def row_with_smallest_domain(self, var: Vars, domains: Domains) -> int:
		""" Find the uninitialized row variabile with the smallest domain. """
		min_domain = len(domains["rows"][var["rows"][0]])
		min_row = var["rows"][0]

		for row in var["rows"][1:]:
			if len(domains["rows"][row]) < min_domain:
				min_domain = len(domains["rows"][row])
				min_row = row
		return min_row

	def bkt_mac(self, board: State, step: int, var: Vars, domains: Domains) -> Optional[State]:
		""" Backtracking algorithm for filling all board's lines, which also
		mantains arc consistency by calling AC3 after each assignment. """
		if step == self.height:
			return deepcopy(board)

		row = self.row_with_smallest_domain(var, domains)
		var["rows"].remove(row)

		self.nodes_expanded += 1

		for val_row in domains["rows"][row]:
			self.nodes_generated += 1

			# Assign value to row, resets its domain to only that value and
			# call AC3 to possibly reduce other domains as well.
			new_domains = deepcopy(domains)
			new_domains["rows"][row] = [val_row]
			if not self.arc_consistency(row, var, new_domains):
				continue
			board[row] = val_row

			ans = self.bkt_mac(board, step + 1, var, new_domains)
			if ans is not None:
				return ans # solution found
			board[row] = [] # undo assignment
		return None

	def solve(self) -> Optional[State]:
		var, domains = self.init_problem()
		return self.bkt_mac([[] for _ in range(self.height)], 0, var, domains)

	def get_strategy_name(self) -> str:
		return "csp"


"""
Utils + main
"""


def parse_args() -> Tuple[str, str]:
	parser = argparse.ArgumentParser(description="Nonograms solver")
	parser.add_argument("fin", help="name of input json file")
	parser.add_argument("fout", help="name of output json file")
	parser.add_argument("-s", "--strategy", default="csp",
		choices=["bfs", "dfs", "iter-deepening", "astar", "csp"],
		help="strategy used for the solver")
	parser.add_argument("--prettyprint", action="store_true")
	args = parser.parse_args()
	return (args.fin, args.fout, args.strategy, args.prettyprint)


def create_solver(strategy: str, data: Any) -> Solver:
	if strategy == "bfs":
		return SolverBFS(**data)
	elif strategy == "dfs":
		return SolverDFS(**data)
	elif strategy == "iter-deepening":
		return SolverIterativeDeepening(**data)
	elif strategy == "astar":
		return SolverAStar(**data)
	else:
		return SolverCSP(**data)


def pretty_print(board: Optional[State]):
	if board is None:
		print("No possible solution")
		return

	for i in range(len(board)):
		print("|".join(list(map(lambda x: "." if x == 0 else "X", board[i]))))


if __name__ == "__main__":
	input_filename, output_filename, strategy, prettyprint = parse_args()

	input_data = None
	with open(input_filename) as fin:
		input_data = json.load(fin)

	solver = create_solver(strategy, input_data)
	solution = solver.solve_to_json()

	with open(output_filename, "w") as fout:
		json.dump(solution, fout, indent=4)

	# Used for testing:
	if prettyprint:
		pretty_print(solution["solution"])
