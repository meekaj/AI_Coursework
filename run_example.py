#to test everything

from env import generate_flmp, FLMPEnv
from astar import astar_search
from agent import follow_path
import numpy as np

def print_grid(desc, path=None):
    #want to see path made clearly...
    path_set = set(path) if path is not None else set()

    for r in range(desc.shape[0]):
        row_symbols = []
        for c in range(desc.shape[1]):
            symbol = desc[r, c]
            #mark path cells * but keep I and G visible
            if (r, c) in path_set and symbol not in ["I", "G"]:
                symbol = "*"
            #width 2 for alignment
            row_symbols.append(f"{symbol:2}")

        print(" ".join(row_symbols))
    print()

if __name__ == "__main__":
    #1. random maze generator
    desc, portals = generate_flmp(size=11, wall_prob=0.2, hole_prob=0.15,
                                            n_portals=2, seed=123)
    env = FLMPEnv(desc=desc, portals=portals, p=1,render_mode="ansi") #change p accordingly...

    #2. show maze grid
    print("Generated Maze:")
    print_grid(desc)
    #3. show portals dictionary
    print("Portals:", portals, "\n") 

    #4. run astar
    path = astar_search(desc, portals)
    print("Path found by A*, shown as a list of (row, col) cells:")
    print (path, "\n")

    if path is not None:
        print("Maze with path marked with *")
        print_grid(desc, path = path)
    else:
        print ("A* could not find path")

    #env = FLMPEnv(desc=desc, portals=portals, render_mode="ansi")
    obs, _ = env.reset()

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
