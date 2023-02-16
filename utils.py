import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_learning_curve(result_dict, plot_file):
    
    scores, steps = result_dict['score_history'], result_dict['step_history']
    running_avg = np.array((pd.DataFrame(scores).rolling(100, min_periods=1).mean())[0])
    fig = plt.figure()
    plt.plot(steps, running_avg, label='100 running avg')
    plt.plot(steps, scores, label='score')
    plt.title('Scores')
    plt.xlabel('Step')
    plt.ylabel('Score per episode')
    plt.legend()
    fig.savefig(plot_file)