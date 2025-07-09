import omni.isaac.lab.envs.mdp as mdp

print("🔍 支持的 reward 函数：")
print([f for f in dir(mdp) if "reward" in f])
