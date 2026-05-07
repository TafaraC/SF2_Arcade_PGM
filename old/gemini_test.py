import stable_retro
import pandas as pd

env = stable_retro.make(game='StreetFighterIISpecialChampionEdition-Genesis-v0', use_restricted_actions=stable_retro.Actions.FILTERED)
data = []


obs = env.reset()
done = False
i = 0
print(env.buttons)
