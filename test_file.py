import pickle

with open("results/experiments/periodic_hills_csep_v1/optimizer.pkl", "rb") as f:
    optimizer = pickle.load(f)

# Now you can inspect it:
print(f"Number of trials done: {len(optimizer.Xi)}")
print(f"Parameter vectors tried: {optimizer.Xi}")
print(f"Scores observed: {optimizer.yi}")
print(f"Best score: {min(optimizer.yi)}")
print(f"Best params: {optimizer.Xi[optimizer.yi.index(min(optimizer.yi))]}")