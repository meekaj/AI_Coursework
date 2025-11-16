import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class FLMPEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, desc, portals=None, p=1.0, render_mode="ansi"):
        super().__init__()
        self.desc = np.asarray(desc, dtype="O") #desc is grid, shape give height and width
        self.nrow, self.ncol = self.desc.shape
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)  # Left, Down, Right, Up
        self.observation_space = spaces.Discrete(self.nrow * self.ncol)

        self.p = p # probability for direction

        self.portals = portals if portals else {} #portal handling
        self._portal_states = {
            self.to_s(r, c): self.to_s(rt, ct)
            for k, (r, c) in self.portals.items()
            if k.startswith("S")
            for eid, (rt, ct) in self.portals.items()
            if eid == "E" + k[1:]
        }

        # Fixed start and goal
        self.start_state = self.to_s(0, 0)  # always top-left
        self.goal_state = self.to_s(self.nrow - 1, self.ncol - 1)  # always bottom-right
        self.s = None

    def to_s(self, row, col):
        return row * self.ncol + col

    def to_row_col(self, s):
        return divmod(s, self.ncol)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = self.start_state
        return self.s, {}

    def step(self, action):
        row, col = self.to_row_col(self.s)
        intended_action = action #what agent is trying to do

        #slipping, using p for intended or else agent slip 90 degrees anti clockwise
        slip_map = {
            0:1,#left slip to down
            1:2,#down slip to right
            2:3, #right slip to up
            3:0#up slip to left
        }

        if np.random.random() < self.p:
            actual_action = action
        else:
            actual_action = slip_map[action]

        if actual_action == 0:   # Left
            new_row, new_col = row, col - 1
        elif actual_action == 1: # Down
            new_row, new_col = row + 1, col
        elif actual_action == 2: # Right
            new_row, new_col = row, col + 1
        else:             # Up
            new_row, new_col = row - 1, col

        # Stay if invalid or wall
        if 0 <= new_row < self.nrow and 0 <= new_col < self.ncol:
            if self.desc[new_row, new_col] != "W":
                row, col = new_row, new_col

        new_state = self.to_s(row, col)
        reward, terminated = 0.0, False

        # Portal teleport
        if new_state in self._portal_states:
            new_state = self._portal_states[new_state]

        r, c = self.to_row_col(new_state)
        cell = self.desc[r, c]

        # Terminal states
        fell_in_hole = False
        if cell == "G":
            reward, terminated = 1.0, True
        elif cell == "H":
            fell_in_hole = True #agent alive but record it fell

        self.s = new_state
        info = {
            "fell_in_hole": fell_in_hole,
            "intended_action": intended_action,
            "actual_action": actual_action,
        }
        return new_state, reward, terminated, False, info

    def render(self): #print new character for each cell
        out = ""
        agent_r, agent_c = self.to_row_col(self.s)
        for r in range(self.nrow):
            for c in range(self.ncol):
                if (r, c) == (agent_r, agent_c):
                    out += "A  "
                elif isinstance(self.desc[r, c], str) and self.desc[r, c].startswith("S"):
                    out += f"{self.desc[r, c]} "
                elif isinstance(self.desc[r, c], str) and self.desc[r, c].startswith("E"):
                    out += f"{self.desc[r, c]} "
                elif self.desc[r, c] == "W":
                    out += "W  "
                elif self.desc[r, c] == "H":
                    out += "H  "
                elif self.desc[r, c] == "G":
                    out += "G  "
                elif self.desc[r, c] == "I":
                    out += "I  "
                else:
                    out += ".  "
            out += "\n"
        if self.render_mode == "ansi":
            return out
        else:
            print(out)

def generate_flmp(size=6, wall_prob=0.2, hole_prob=0.1, n_portals=2, seed=None):
    """Generate a random Frozen Lake Maze with Portals (FLMP).

    Start state = always top-left (0,0).
    Goal state = always bottom-right (size-1, size-1).
    """
    rng = random.Random(seed)
    grid = [["." for _ in range(size)] for _ in range(size)]

    # Fixed start and goal
    init_pos = (0, 0)
    goal_pos = (size - 1, size - 1)
    grid[init_pos[0]][init_pos[1]] = "I"
    grid[goal_pos[0]][goal_pos[1]] = "G"

    # Add walls and holes
    for r in range(size):
        for c in range(size):
            if (r, c) in [init_pos, goal_pos]:
                continue
            roll = rng.random()
            if roll < wall_prob:
                grid[r][c] = "W"
            elif roll < wall_prob + hole_prob:
                grid[r][c] = "H"

    # Place portals
    free_cells = [(r, c) for r in range(size) for c in range(size)
                  if grid[r][c] == "."]
    rng.shuffle(free_cells)

    portals = {}
    for i in range(n_portals):
        if len(free_cells) < 2:
            break
        start, end = free_cells.pop(), free_cells.pop()
        portals[f"S{i}"] = start
        portals[f"E{i}"] = end
        grid[start[0]][start[1]] = f"S{i}"
        grid[end[0]][end[1]] = f"E{i}"

    return np.array(grid), portals
