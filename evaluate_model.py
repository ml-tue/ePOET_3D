from .poet_distributed.niches.box2d.model import Model, simulate #, novelty_vec
from .poet_distributed.niches.box2d.env import env_custom, Env_config
from .poet_distributed.niches.box2d.cppn import CppnEnvParams
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from gym.wrappers import Monitor
# from .poet_distributed.novelty import compute_behavior_novelty_vs_archive

## render the trained model
def render_results():
    directory1 = './trained_models/'
    env_config = Env_config(
            name='default_env',
            ground_roughness=0.5,
            pit_gap=[],
            stump_width=[],
            stump_height=[],
            stump_float=[],
            stair_height=[],
            stair_width=[],
            stair_steps=[])
    total_novelty = []
    model = Model(env_custom)
    model.make_env('flat1', render_mode=True, env_config=env_config)

    novelty_vec = []
    levels = []
    finished = []
    actions_list = []
    rewards_list = []
    for theta_file in sorted(os.listdir(directory1 + 'poet_sac/poet_twin22/')):
        if theta_file.endswith(".json"):
            theta = directory1 + 'poet_sac/poet_twin22/' + theta_file
            model.load_model(theta)
            rs_seeds = np.random.randint(np.int32(2 ** 31 - 1), size=256)
            random_state = np.random.RandomState(rs_seeds).randint(1000000)
            for cppngenome in sorted(os.listdir(directory1 + 'poet_sac/cppngenomes_admitted_twin/')):
                print('run with theta {} on envrionment {}'.format(theta_file, cppngenome))
                returns, lengths, novelty, level, finish, action_list = simulate(
                    model, seed=random_state, render_mode=False, train_mode=False, num_episode=1,
                    env_params=CppnEnvParams(
                        genome_path=directory1 + 'poet_sac/cppngenomes_admitted_twin/' + cppngenome))
                levels.append(level)
                finished.append(finish)
                actions_list.append(action_list[0])
                rewards_list.append(returns[0])
                total_novelty.append(novelty)
                print('return: ', returns)
                novelties = []
                novelties.append(np.mean(total_novelty, axis=0))
                novelty_vec.append(np.mean(novelties,axis=0))
    return levels, finished, actions_list, rewards_list

## analyze the number of solved environments
def env_ana(finished, levels):
    print(finished)
    print(levels)
    finished = np.array(finished)
    levels = np.array(levels)
    solved = np.count_nonzero(finished==True)
    print('solved {} environments out of {}'.format(solved, len(finished)))
    easy = np.count_nonzero(levels < 0.001)
    middle = np.count_nonzero(np.logical_and(levels >= 0.001, levels < 0.01))
    hard = np.count_nonzero(np.logical_and(levels >= 0.01, levels < 0.1))
    exetreme = np.count_nonzero(levels >= 0.1)
    solved_list = np.where(finished==True)[0]
    easy_s = np.count_nonzero(np.isin(solved_list,np.where(levels < 0.001)[0]))
    middle_s = np.count_nonzero(np.isin(solved_list,np.where(np.logical_and(levels >= 0.001, levels < 0.01))[0]))
    hard_s = np.count_nonzero(np.isin(solved_list,np.where(np.logical_and(levels >= 0.01, levels < 0.1))[0]))
    exetreme_s = np.count_nonzero(np.isin(solved_list,np.where(levels >= 0.1)[0]))
    Envs = {'Easy':easy, 'Middle':middle, 'Hard':hard, 'Exetreme':exetreme, 'Total': len(finished)}
    Envs_solv = {'Easy': easy_s, 'Middle': middle_s, 'Hard': hard_s, 'Exetreme': exetreme_s, 'Total': solved}

    env_level = list(Envs.keys())
    level_values = list(Envs.values())
    solved_values = list(Envs_solv.values())
    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=(10, 5))  color='maroon'
    ax.bar(env_level, level_values, width=0.4, label='generated')
    ax.bar(env_level, solved_values, width=0.4, label='solved')

    heights = []
    for i, rec in enumerate(ax.patches):
        if math.floor(i/len(level_values))==1:
            heights.append(rec.get_height())
    for i, rec in enumerate(ax.patches):
        height = rec.get_height()
        if math.floor(i/len(level_values))==0 and height>0:
            ax.text(rec.get_x() + rec.get_width() / 2,
                      rec.get_y() + height,
                      "{:.0f}%".format((heights[i]/height)*100),
                      ha='center',
                      va='bottom')
    ax.set_xlabel('Level of generated environment')
    ax.set_ylabel('Counts')
    ax.legend()
    plt.show()

## heatmap of environment-agent pairs
def heatmap(finished):
    finished = np.array(finished).reshape((int(np.sqrt(len(finished))),int(np.sqrt(len(finished)))))
    ax = sns.heatmap(finished, cmap='YlGnBu', linewidths=0.1, annot=True, cbar=False)
    ax.set_xlabel('Env')
    ax.set_ylabel('Optimizer')
    plt.title('POET-SAC solved env-agent pairs')
    plt.show()

## analyze the action representation
def action_ana(actions_list):
    # tsne = TSNE(random_state=1, n_iter=15000, metric="cosine")
    tsne = TSNE(random_state=123, n_iter=15000, metric="euclidean")

    FS = (10, 8)
    fig, ax = plt.subplots(figsize=FS)
    for i in range(np.array(actions_list).shape[0]):
        embs = tsne.fit_transform(actions_list[i])
        df = pd.DataFrame()
        df['x'] = embs[:, 0]
        df['y'] = embs[:, 1]
        ax.scatter(df.x, df.y, alpha=.1)
    plt.show()

if __name__ == '__main__':
    ## evaluate the trained models
    levels, finished, actions_list, rewards = render_results()
    # print(rewards)
    # print(levels)
    # print(finished)
    # print(actions_list)

    ## analyze the number of solved environments
    ## if not then comment out this function
    env_ana(finished, levels)

    ## heatmap of environment-agent pairs
    ## if not then comment out this function
    heatmap(finished)

    ## analyze the action representation
    ## if not then comment out this function
    ## actins_list of all runs will be extreme large, so better to only run for less than five environments
    action_ana(actions_list)
