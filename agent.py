from env import FLMPEnv, generate_flmp
from astar import astar_search


def action_towards(current, target):
    #current (r, c) and target (tr, tc), return an action that moves towards target ish
    (r, c) = current
    (tr, tc) = target
    dr = tr - r
    dc = tc - c

    #at the target, dont move
    if dr == 0 and dc == 0:
        return None

    #moving along the axis where the distance is larger
    if abs(dr) > abs(dc):
        #vertically
        if dr > 0:
            return 1  # DOWN
        else:
            return 3  # UP
    else:
        #horizontally
        if dc > 0:
            return 2  # RIGHT
        else:
            return 0  # LEFT


def follow_path(env, path, max_steps=200, verbose=False):
    """
    let agent follow the a star path in slippery environment.

    to return if successful, how many steps and how many times it fell
    """
    if path is None or len(path) == 0:
        raise ValueError("Path is None or empty; cannot follow it.")
    
    obs, _ = env.reset()
    steps = 0
    hole_falls = 0

    goal_state = env.goal_state

    #map from state index to (row, col)
    def state_to_rc(s):
        return env.to_row_col(s)

    for t in range(max_steps):
        r, c = state_to_rc(obs)

        # if we reached the goal, stop
        if obs == goal_state:
            if verbose:
                print(f"Reached goal at step {steps}.")
            return True, steps, hole_falls

        # find the closest point on the A* path to our current position using Manhattan distance
        distances = [abs(pr - r) + abs(pc - c) for (pr, pc) in path]
        closest_idx = min(range(len(path)), key=lambda i: distances[i])

        # choose a target: one step ahead along the path if possible
        if closest_idx < len(path) - 1:
            target = path[closest_idx + 1]
        else:
            target = path[closest_idx]  # already at the goal cell

        action = action_towards((r, c), target)
        if action is None:
            # already exactly at target, nothing to do
            steps +=1
            continue

        obs, reward, terminated, truncated, info = env.step(action)
        steps +=1

        if info.get("fell_in_hole", False):
            hole_falls += 1

        if verbose:
            nr, nc = state_to_rc(obs)
            print(
                f"Step {steps}: intended={info['intended_action']}, "
                f"actual={info['actual_action']}, "
                f"pos=({nr}, {nc}), target={target}, "
                f"reward={reward}, fell_in_hole={info['fell_in_hole']}"
            )

        if terminated or truncated:
            success = (reward > 0)
            return success, steps, hole_falls
    return False, steps, hole_falls #we did not reach goal in time

