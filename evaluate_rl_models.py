import numpy as np
import torch
from torchrl.utils import get_args
from torchrl.utils import get_params
from torchrl import policies
from torchrl import networks
from torchrl.env import get_vec_env
from torchrl.algo import PPO
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm

def pre_env_ana(finished, levels):
    finished = np.array(finished)
    levels = np.array(levels)
    solved = np.count_nonzero(finished == True)
    print('solved {} environments out of {}'.format(solved, len(finished)))
    # easy = np.count_nonzero(levels < 0.001)
    # middle = np.count_nonzero(np.logical_and(levels >= 0.001, levels < 0.01))
    # hard = np.count_nonzero(np.logical_and(levels >= 0.01, levels < 0.1))
    # exetreme = np.count_nonzero(levels >= 0.1)
    easy = 9
    middle = 14
    hard = 9
    exetreme = 0
    # print(easy, middle, hard, exetreme)
    solved_list = np.where(finished == True)[0]
    easy_s = np.count_nonzero(np.isin(solved_list, np.where(levels < 0.001)[0]))
    middle_s = np.count_nonzero(np.isin(solved_list, np.where(np.logical_and(levels >= 0.001, levels < 0.01))[0]))
    hard_s = np.count_nonzero(np.isin(solved_list, np.where(np.logical_and(levels >= 0.01, levels < 0.1))[0]))
    exetreme_s = np.count_nonzero(np.isin(solved_list, np.where(levels >= 0.1)[0]))
    Envs = {'Easy': easy, 'Middle': middle, 'Hard': hard, 'Extreme': exetreme, 'Total': len(finished)}
    Envs_solv = {'Easy': easy_s, 'Middle': middle_s, 'Hard': hard_s, 'Extreme': exetreme_s, 'Total': solved}

    env_level = list(Envs.keys())
    level_values = list(Envs.values())
    solved_values = list(Envs_solv.values())
    return env_level, level_values, solved_values

def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot", H="/", **kwargs):
    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)
    for df in dfall:  # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h, l = axe.get_legend_handles_labels()  # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i + n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col))  # edited part
                rect.set_width(1 / float(n_df + 1))
    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation=0)
    axe.set_title(title)
    n = []
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))
    l1 = axe.legend(h[:n_col], l[:n_col], loc=[0.2, 0.85])
    if labels is not None:
        l2 = plt.legend(n, labels) #, loc=[1.01, 0.1]
    axe.add_artist(l1)
    plt.show()
    return axe

def plot_env_ana(finished_ppo, levels_ppo, finished_sac, levels_sac, finished_poet, levels_poet):
    env_level_ppo, level_values_ppo, solved_values_ppo = pre_env_ana(finished_ppo, levels_ppo)
    env_level_sac, level_values_sac, solved_values_sac = pre_env_ana(finished_sac, levels_sac)
    env_level_poet, level_values_poet, solved_values_poet = pre_env_ana(finished_poet, levels_poet)
    print(np.stack(np.array([level_values_ppo, solved_values_ppo]), axis=1))
    df_ppo = pd.DataFrame(np.stack(np.array([level_values_ppo, solved_values_ppo]), axis=1),
                       index=["Easy", "Middle", "Hard", "Extreme", "Total"],
                       columns=["Solved", "Generated"])
    df_sac = pd.DataFrame(np.stack(np.array([level_values_sac, solved_values_sac]), axis=1),
                          index=["Easy", "Middle", "Hard", "Extreme", "Total"],
                          columns=["Solved", "Generated"])
    df_poet = pd.DataFrame(np.stack(np.array([level_values_poet, solved_values_poet]), axis=1),
                          index=["Easy", "Middle", "Hard", "Extreme", "Total"],
                          columns=["Solved", "Generated"])
    plot_clustered_stacked([df_ppo, df_sac, df_poet], ["ppo", "sac", "poet"])

def plot_rewards(rewards_poet, rewards_poet_nov, rewards_poet_sac):
    x = [i for i in range(len(rewards_poet[10:]))]
    plt.plot(x, np.sort(np.array(rewards_poet))[10:], label='poet')
    # plt.plot(x, np.sort(np.array(rewards_sac))[10:], label='sac')
    # plt.plot(x, np.sort(np.array(rewards_ppo))[10:], label='ppo')
    plt.plot(x, np.sort(np.array(rewards_poet_nov))[10:], label='poet_noveltyS')
    plt.plot(x, np.sort(np.array(rewards_poet_sac))[10:], label='poet-sac')
    plt.xlabel('Environment')
    plt.ylabel('Evaluation return')
    plt.title('Evaluation returns in generated environments')
    plt.legend()
    plt.show()

def render_result(config_file, alg, model_file):
    args = get_args()
    params = get_params(config_file)
    device = torch.device('cpu')
    env = get_vec_env(
            params["env_name"],
            params["env"],
            vec_env_nums=1
        )

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    np.random.seed(args.seed)

    params['general_setting']['env'] = env
    params['net']['base_type'] = networks.MLPBase
    if alg == 'PPO':
        params['net']['activation_func'] = torch.nn.Tanh
        pf = policies.GuassianContPolicyBasicBias(
                input_shape=env.observation_space.shape[0],
                output_shape=env.action_space.shape[0],
                **params['net'],
                **params['policy']
            )
    elif alg == 'SAC' or alg == 'SAC_sensor':
        params['net']['activation_func'] = torch.nn.ReLU
        pf = policies.GuassianContPolicy(
            input_shape=env.observation_space.shape[0],
            output_shape=2 * env.action_space.shape[0],
            **params['net'],
            **params['policy'])
    elif alg == 'TD3':
        params['net']['activation_func'] = torch.nn.ReLU
        pf = policies.FixGuassianContPolicy(
            input_shape=env.observation_space.shape[0],
            output_shape=env.action_space.shape[0],
            **params['net'],
            **params['policy'])

    # vf = networks.Net(
    #     input_shape=env.observation_space.shape,
    #     output_shape=1,
    #     **params['net']
    # )
    # vf_path = '/home/fang/project/thesis/torchrl/log/ppo-hex/Ant-v0/0/model/model_vf_best.pth'
    # vf.load_state_dict(torch.load(vf_path, map_location=device))

    pf.load_state_dict(torch.load(model_file, map_location=device))

    import os
    from .poet_distributed.niches.box2d.cppn import CppnEnvParams
    cppn_dir = './trained_models/cppngenomes/cppngenomes_b/'
    levels = []
    finished = []
    eval_rewards = []
    actions_list = []
    ind = 0
    for cppngenome in os.listdir(cppn_dir):
        print('run on envrionment {}'.format(cppngenome))
        env_params = CppnEnvParams(genome_path=cppn_dir + cppngenome)
        env.augment(env_params)

        obs = env.reset()
        total_reward = 0.0
        avarage_var_height = 0.0
        ts = 0
        ind += 1
        action_list = []
        for t in range(5000):
            env.render("human")
            action = pf.eval_act(torch.Tensor(obs).to(device).unsqueeze(0))
            obs, reward, done, info = env.step(action)
            total_reward += reward
            avarage_var_height += info['level']
            ts += 1
            if done:
                break
            action_list.append(action.tolist()[0])
        # print('total_reward', total_reward)
        avarage_var_height /= min(ts, 5000)
        levels.append(avarage_var_height[0])
        finished.append(info['finish'][0])
        eval_rewards.append(total_reward[0][0])
        actions_list.append(action_list)
        # print('total reward: {}, level: {}, finished: {}'.format(total_reward, avarage_var_height, info['finish']))
        # if ind > 2:
        #     break
    return levels, finished, eval_rewards, actions_list

def action_ana(actions_list):
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
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

def env_ana(finished, levels, alg):
    env_level_ppo, level_values_ppo, solved_values_ppo = pre_env_ana(finished, levels)
    # env_level_ppo, level_values_ppo, solved_values_ppo = pre_env_ana(finished_ppo, levels_ppo)
    # env_level_sac, level_values_sac, solved_values_sac = pre_env_ana(finished_sac, levels_sac)
    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=(10, 5))  color='maroon'
    ax.bar(env_level_ppo, level_values_ppo, width=0.3, label='generated')
    ax.bar(env_level_ppo, solved_values_ppo, width=0.3, label='solved')
    # ax.bar(env_level_sac, level_values_sac, width=0.4, label='generated')

    heights = []
    import math
    for i, rec in enumerate(ax.patches):
        if math.floor(i/len(level_values_ppo))==1:
            heights.append(rec.get_height())
    for i, rec in enumerate(ax.patches):
        height = rec.get_height()
        if math.floor(i/len(level_values_ppo))==0 and height>0:
            ax.text(rec.get_x() + rec.get_width() / 2,
                      rec.get_y() + height,
                      "{:.0f}%".format((heights[i]/height)*100),
                      ha='center',
                      va='bottom')
    ax.set_xlabel('Level of generated environment')
    ax.set_ylabel('Counts')
    ax.set_title('Solved rate by ' + alg)
    ax.legend()
    plt.show()

ppo_config = './torchrl/torchrl/config/ppo_hex.json'
twin_sac_config = './torchrl/torchrl/config/twin_sac_q_hex.json'
td3_config = './torchrl/torchrl/config/td3_hex.json'
ppo_pf = './trained_models/ppo/0/model/model_pf_best.pth'
twin_sac_pf = './trained_models/sac/42/model/model_pf_best.pth'
td3_pf = './trained_model/td3/42/model/model_pf_best.pth'
def renders(alg):
    if alg == 'SAC':
        config_file = twin_sac_config
        model_file = twin_sac_pf
    elif alg == 'PPO':
        config_file = ppo_config
        model_file = ppo_pf
    elif alg == 'TD3':
        config_file = ppo_pf
        model_file = td3_pf
    levels, finished, eval_rewards, actions_list = render_result(config_file=config_file, alg=alg, model_file=model_file)
    # print(levels, finished, eval_rewards)
    return levels, finished, eval_rewards, actions_list

if __name__ == '__main__':
    ## input algorithm: PPO, SAC, TD3
    levels, finished, eval_rewards, actions_list = renders('SAC')

    env_ana(finished, levels, 'SAC')

    # plot_env_ana(finished_ppo, levels_ppo, finished_sac, levels_sac, finished_poet, levels_poet)

    ## plot action representation, better less than 5 environments
    action_ana(actions_list)

    plot_rewards()

