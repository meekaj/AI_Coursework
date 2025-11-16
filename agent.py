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
    obs, _ = env.reset()
    steps = 0
    hole_falls = 0

    nrow, ncol = env.nrow, env.ncol
    goal_state = env.goal_state

    #map from state index to (row, col)
    def state_to_rc(s):
        return env.to_row_col(s)

    #(start at the first cell)
    idx = 0

    for t in range(max_steps):
        steps += 1

        r, c = state_to_rc(obs)

        #if we reached the goal, stop
        if obs == goal_state:
            if verbose:
                print(f"Reached goal at step {steps}.")
            return True, steps, hole_falls

        #choose target on the path we want to move towards
        if idx < len(path) - 1 and (r, c) == path[idx]:
            # move to the next point on path
            target = path[idx + 1]
            idx += 1
        else:
            # if we've slipped off, aim for goal
            target = path[-1]

        action = action_towards((r, c), target)
        if action is None:
            # already at target, just continue
            continue

        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("fell_in_hole", False):
            hole_falls += 1

        if verbose:
            nr, nc = state_to_rc(obs)
            print(
                f"Step {steps}: intended={info['intended_action']}, "
                f"actual={info['actual_action']}, "
                f"pos=({nr}, {nc}), reward={reward}, "
                f"fell_in_hole={info['fell_in_hole']}"
            )

        if terminated or truncated:
            #if terminated by reaching goal, success = True
            success = (reward > 0)
            return success, steps, hole_falls

    # did not reach goal within max_steps
    return False, steps, hole_falls
