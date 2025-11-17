#to test everything

from env import generate_flmp, FLMPEnv
from astar import astar_search
from agent import follow_path
import numpy as np

def print_grid(desc, path=None, agent_pos=None):
    RESET  = "\033[0m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"

    path_set = set(path) if path is not None else set()

    for r in range(desc.shape[0]):
        row_symbols = []
        for c in range(desc.shape[1]):
            symbol = desc[r, c]

            if agent_pos is not None and (r, c) == agent_pos:
                vis = "A"
            elif (r, c) in path_set and symbol not in ["I", "G"]:
                vis = "*"
            else:
                vis = symbol 

            cell = f"{vis:2}"

            #colour
            if agent_pos is not None and (r, c) == agent_pos:
                cell = f"{RED}{cell}{RESET}"
            elif (r, c) in path_set and symbol not in ["I", "G"]:
                cell = f"{YELLOW}{cell}{RESET}"
            elif symbol == "G":
                cell = f"{GREEN}{cell}{RESET}"

            row_symbols.append(cell)

        print(" ".join(row_symbols))
    print()


if __name__ == "__main__":
    #1. random maze generator
    desc, portals = generate_flmp(size=12, wall_prob=0.2, hole_prob=0.15,
                                            n_portals=2)
    env = FLMPEnv(desc=desc, portals=portals, p=1,render_mode="ansi") #change p accordingly...

    obs, _ = env.reset()
    agent_r, agent_c = env.to_row_col(obs)

    #2. show maze grid
    print("Generated Maze:")
    print_grid(desc, agent_pos=(agent_r, agent_c))
    #3. show portals dictionary
    print("Portals:", portals, "\n") 

    #4. run astar
    path = astar_search(desc, portals)
    print("Path found by A*, shown as a list of (row, col) cells:")
    print (path, "\n")

    

    if path is not None:
        print("Maze with path marked with *")
        print_grid(desc, path=path, agent_pos=(agent_r, agent_c))

    #env = FLMPEnv(desc=desc, portals=portals, render_mode="ansi")
    
    #print("inital env")
    #print(env.render())
    
    if path is not None:
        env = FLMPEnv(desc=desc, portals=portals, p=1, render_mode="ansi")
        success, steps, hole_falls = follow_path(env, path, max_steps=300, verbose=True)
        print("Follow-path run summary:")
        print(f"  Success: {success}")
        print(f"  Steps taken: {steps}")
        print(f"  Hole falls: {hole_falls}")
        env.close()
    else:
        print("A* found no path.")

    '''    
    done = False
    while not done:
        #print(env.render())
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        row, col = env.to_row_col(obs)
        print(f"Action: {action}, State: {obs} (row={row}, col={col}), Reward: {reward}, fell_in_hole={info['fell_in_hole']}")
    '''

    env.close()
