import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

log = pkl.load(open("train_log.pkl", "rb"))
print(log)