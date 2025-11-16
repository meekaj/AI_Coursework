#to test everything

# Example usage
if __name__ == "__main__":
    desc, portals = generate_flmp(size=12, wall_prob=0.2, hole_prob=0.15,
                                            n_portals=2, seed=123)
    env = FLMPEnv(desc=desc, portals=portals, render_mode="ansi")

#     print("Generated Maze:")
#     for row in desc:
#         print(" ".join(row))
#     print("Portals:", portals)

    obs, _ = env.reset()
    print(env.render())

#     done = False
#     while not done:
#         print(env.render())
#         action = env.action_space.sample()
#         obs, reward, done, _, info = env.step(action)
#         print(f"Action: {action}, State: {obs}, Reward: {reward}")
    env.close()
