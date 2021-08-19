import pandas as pd
import matplotlib.pyplot as plt
import os

directory_poet = './trained_models/poet/poet_hexA8/' ## trained with threshold=2000
directory2 = './trained_models/poet/poet_hexA807/' ## local optimum with threshold=1400
directory_nov = './trained_models/poet_ns/poet_nov8/'
directory_twin = './trained_models/poet_sac/poet_twin22/'
d_ppo = './trained_models/ppo/0/log.csv'
d_vmpo = './trained_models/vmpo/42/log.csv'
d_twin_sac_q = './trained_models/sac/42/log.csv'
# d_td3 = './trained_models/td3/42/log.csv'
# d_a2c = '/home/fang/project/thesis/torchrl/log/a2c_hex/Hexapod/42/log.csv'

def plot_eval(experiment, terminate=False, poet_sac=False, threshold=False):
    if experiment == 'ppo-sac-poet':
        ## load ppo log
        df_ppo_read = pd.read_csv(d_ppo)
        df_ppo = pd.DataFrame()
        # print(df_ppo_read.columns) #Running_Average_Rewards, Train_Epoch_Reward, Running_Training_Average_Rewards,
        df_ppo['iteration'] = df_ppo_read['EPOCH']
        df_ppo['Train_Epoch_Reward'] = df_ppo_read['Train_Epoch_Reward']
        # df_ppo['Train_Epoch_Reward'] = df_ppo_read['Running_Training_Average_Rewards']

        ## load vmpo log
        df_vmpo_read = pd.read_csv(d_vmpo)
        df_vmpo = pd.DataFrame()
        df_vmpo['iteration'] = df_vmpo_read['EPOCH']
        df_vmpo['Train_Epoch_Reward'] = df_vmpo_read['Train_Epoch_Reward']

        ## load sac log
        df_twin_sac_q_read = pd.read_csv(d_twin_sac_q)
        df_twin_sac_q = pd.DataFrame()
        df_twin_sac_q['iteration'] = df_twin_sac_q_read['EPOCH']
        df_twin_sac_q['Train_Epoch_Reward'] = df_twin_sac_q_read['Train_Epoch_Reward']

        # ## load td3 log
        # df_td3_read = pd.read_csv(d_td3)
        # df_td3 = pd.DataFrame()
        # df_td3['iteration'] = df_td3_read['EPOCH']
        # df_td3['Train_Epoch_Reward'] = df_td3_read['Train_Epoch_Reward']

        # ## load a2c log
        # df_a2c_read = pd.read_csv(d_a2c)
        # df_a2c = pd.DataFrame()
        # df_a2c['iteration'] = df_a2c_read['EPOCH']
        # df_a2c['Train_Epoch_Reward'] = df_a2c_read['Train_Epoch_Reward']

        plt.plot(df_ppo['iteration'], df_ppo['Train_Epoch_Reward'], label='ppo')
        plt.plot(df_vmpo['iteration'], df_vmpo['Train_Epoch_Reward'], label='vmpo')
        plt.plot(df_twin_sac_q['iteration'], df_twin_sac_q['Train_Epoch_Reward'], label='twin_sac_q')
        # plt.plot(df_td3['iteration'], df_td3['Train_Epoch_Reward'], label='td3')
        # plt.plot(df_a2c['iteration'], df_a2c['Train_Epoch_Reward'], label='a2c')

    elif experiment == "poet-ns-sac":
        if poet_sac == False and threshold == False:
            ## poet + noverlty search plot
            df2_nov = pd.DataFrame()
            df_i_nov = pd.DataFrame()
            for log_file_nov in os.listdir(directory_nov):
                if log_file_nov.endswith(".log"):
                    file_path_nov = directory_nov + log_file_nov
                    df_nov = pd.read_csv(file_path_nov)

                    if df_nov.columns[0] == 'po_returns_mean_flat':
                        df_i_nov['iteration'] = df_nov['iteration']
                    df_tmp_nov = df_nov[df_nov.columns[9]].copy()
                    df2_nov = pd.concat([df2_nov, df_tmp_nov], axis=1, join="outer")
            df2_nov = pd.concat([df2_nov, df_i_nov], axis=1, join="outer")
            j = 0
            for column in df2_nov.columns:
                if column != 'iteration':
                    if column == 'eval_returns_mean_flat':
                        # label = column
                        label = 'poet_noveltyS_gen0'
                        # plt.plot(df2_nov['iteration'], df2_nov[column], label=label)
                    else:
                        j += 1
                        label = 'poet_noveltyS_gen' + str(j)
                    plt.plot(df2_nov['iteration'], df2_nov[column], label=label)

        ## poet-sac plot
        df2_twin = pd.DataFrame()
        df_i_twin = pd.DataFrame()
        if threshold == True:
            directory_twin = './trained_models/poet_sac/poet_twin24/'
        else:
            directory_twin = './trained_models/poet_sac/poet_twin22/'
        for log_file_twin in os.listdir(directory_twin):
            if log_file_twin.endswith(".log"):
                file_path_twin = directory_twin + log_file_twin
                df_twin = pd.read_csv(file_path_twin)

                if df_twin.columns[0] == 'po_returns_mean_flat':
                    df_i_twin['iteration'] = df_twin['iteration']
                df_tmp_twin = df_twin[df_twin.columns[9]].copy()
                df2_twin = pd.concat([df2_twin, df_tmp_twin], axis=1, join="outer")
        df2_twin = pd.concat([df2_twin, df_i_twin], axis=1, join="outer")
        m = 0
        df2_twin = df2_twin.reindex(columns=['eval_returns_mean_flat', 'eval_returns_mean_sac_58', 'eval_returns_mean_sac_118',
                                             'eval_returns_mean_e6cf5d0d-0663-4ad7-929e-ef7f84b7282b', 'eval_returns_mean_sac_178',
                                             'eval_returns_mean_35f145c3-0104-431b-96b1-37a357131035', 'eval_returns_mean_sac_238', 'iteration'])
        for column in df2_twin.columns:
            if column != 'iteration':
                if column == 'eval_returns_mean_flat':
                    # label = column
                    label = 'poet-sac_gen0'
                    if terminate == True:
                        plt.plot(df2_twin['iteration'], df2_twin[column].shift(periods=df2_twin[column].isna().sum()), label=label)
                        break
                    # plt.plot(df2_twin['iteration'], df2_twin[column], label=label)
                else:
                    m += 1
                    label = 'poet-sac_gen' + str(m)
                # plt.plot(df2_twin['iteration'], df2_twin[column], label=label)
                plt.plot(df2_twin['iteration'], df2_twin[column].shift(periods=df2_twin[column].isna().sum()), label=label)
                if m>0 and threshold==True:
                    break

    if threshold == True:
        ## poet with threshold=1400 plot
        df2_poet2 = pd.DataFrame()
        df_i_poet2 = pd.DataFrame()
        for log_file_poet2 in os.listdir(directory2):
            if log_file_poet2.endswith(".log"):
                file_path_poet2 = directory2 + log_file_poet2
                df_poet2 = pd.read_csv(file_path_poet2)

                if df_poet2.columns[0] == 'po_returns_mean_flat':
                    df_i_poet2['iteration'] = df_poet2['iteration']
                df_tmp_poet2 = df_poet2[df_poet2.columns[9]].copy()
                df2_poet2 = pd.concat([df2_poet2, df_tmp_poet2], axis=1, join="outer")
        df2_poet2 = pd.concat([df2_poet2, df_i_poet2], axis=1, join="outer")
        n = 0
        df2_poet2 = df2_poet2.reindex(
            columns=['eval_returns_mean_flat', 'eval_returns_mean_sac_58', 'eval_returns_mean_sac_118',
                     'eval_returns_mean_e6cf5d0d-0663-4ad7-929e-ef7f84b7282b', 'eval_returns_mean_sac_178',
                     'eval_returns_mean_35f145c3-0104-431b-96b1-37a357131035', 'eval_returns_mean_sac_238', 'iteration'])
        for column in df2_poet2.columns:
            if column != 'iteration':
                if column == 'eval_returns_mean_flat':
                    label = 'poet-1400_gen0'
                    plt.plot(df2_poet2['iteration'], df2_poet2[column].shift(periods=df2_poet2[column].isna().sum()), label=label)
                else:
                    n += 1
                    label = 'poet-1400_gen' + str(n)
                # plt.plot(df2_poet2['iteration'], df2_poet2[column].shift(periods=df2_poet2[column].isna().sum()), label=label)
                if n>0:
                    break

    ## poet with threshold=2000 plot
    if threshold == True:
        directory_poet = './trained_models/poet/poet_hexA29/'  ## local optimium with threshold==2000
    else:
        directory_poet = './trained_models/poet/poet_hexA8/'
    if poet_sac == False:
        df2 = pd.DataFrame()
        df_i = pd.DataFrame()
        for log_file in os.listdir(directory_poet):
            if log_file.endswith(".log"):
                file_path = directory_poet + log_file
                df = pd.read_csv(file_path)

                if df.columns[0] == 'po_returns_mean_flat':
                    df_i['iteration'] = df['iteration']
                df_tmp = df[df.columns[9]].copy()
                df2 = pd.concat([df2, df_tmp], axis=1, join="outer")

        df2 = pd.concat([df2, df_i], axis=1, join="outer")
        i = 0
        # dd = df2.reindex(columns=sorted(df2.columns))
        # dd = pd.Index([df2[column].count() for column in df2.columns])
        # dd = dd.sort_values(ascending=False, return_indexer=True)
        # df2 = df2.sort_index(axis=1, level=dd[1])
        df2 = df2.reindex(columns=['eval_returns_mean_flat', 'eval_returns_mean_ec3e1807-f015-4258-87bd-16c8352d0c33',
                                   'eval_returns_mean_c30dce29-c18e-4e1d-bad9-676a2bbe73cf', 'eval_returns_mean_dc7b1c98-c850-4529-bcbf-75119c7c93e9',
                                   'eval_returns_mean_e3653398-afec-4958-ac16-5d867338ab64', 'iteration'])
        for column in df2.columns:
            if column != 'iteration':
                if column == 'eval_returns_mean_flat':
                    # label = column
                    label = 'poet_gen0'
                    # # plt.plot(df2['iteration'], df2[column], label=label)
                    if terminate == True:
                        plt.plot(df2['iteration'], df2[column].shift(periods=df2[column].isna().sum()), label=label)
                        break
                else:
                    i += 1
                    label = 'poet_gen' + str(i)
                plt.plot(df2['iteration'], df2[column].shift(periods=df2[column].isna().sum()), label=label)
                # # plt.plot(df2['iteration'], df2[column], label=label)
                if i>0 and threshold==True:
                    break

    plt.xlabel('iteration')
    plt.ylabel('returns_mean')
    plt.title('Returns of Hexapod')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    ## comparison of ppo, sac, and poet
    plot_eval('ppo-sac-poet')

    ## comparison of poet, poet with novelty search, poet with sac
    plot_eval('poet-ns-sac', terminate=True, poet_sac=False)

    ## plot only poet-sac
    plot_eval('poet-ns-sac', terminate=False, poet_sac=True)

    # plot local convergence of poet
    plot_eval('poet-ns-sac', threshold=True)

'''
Index(['po_returns_mean_flat', 'po_returns_median_flat', 'po_returns_std_flat',
       'po_returns_max_flat', 'po_returns_min_flat', 'po_len_mean_flat',
       'po_len_std_flat', 'noise_std_flat', 'learning_rate_flat',
       'eval_returns_mean_flat', 'eval_returns_median_flat',
       'eval_returns_std_flat', 'eval_len_mean_flat', 'eval_len_std_flat',
       'eval_n_episodes_flat', 'theta_norm_flat', 'grad_norm_flat',
       'update_ratio_flat', 'episodes_this_step_flat', 'episodes_so_far_flat',
       'timesteps_this_step_flat', 'timesteps_so_far_flat',
       'time_elapsed_this_step_flat', 'accept_theta_in_flat',
       'eval_returns_mean_best_in_flat',
       'eval_returns_mean_best_with_ckpt_in_flat',
       'eval_returns_mean_theta_from_others_in_flat',
       'eval_returns_mean_proposal_from_others_in_flat', 'time_elapsed_so_far',
       'iteration'],
      dtype='object')
'''

def plot_po_eval():
    for log_file in os.listdir(directory_poet): #poet_ant1
        if log_file.endswith(".log"):
            file_path = directory_poet + log_file
            df = pd.read_csv(file_path)
            plt.plot(df['iteration'], df[df.columns[0]], c='red', label='po')
            plt.plot(df['iteration'], df[df.columns[9]], c='blue', label='eval')
            plt.xlabel('iteration')
            plt.ylabel('returns_mean')
            plt.title(log_file)
            plt.legend()
            plt.show()

def plot_po():
    df2 = pd.DataFrame()
    df_i = pd.DataFrame()
    for log_file in os.listdir(directory_poet):
        if log_file.endswith(".log"):
            file_path = directory_poet + log_file
            df = pd.read_csv(file_path)

            if df.columns[0] == 'po_returns_mean_flat':
                df_i['iteration'] = df['iteration']
            df_tmp = df[df.columns[0]].copy()
            df2 = pd.concat([df2, df_tmp], axis=1, join="outer")

    df2 = pd.concat([df2, df_i], axis=1, join="outer")
    i = 0
    for column in df2.columns:
        if column != 'iteration':
            if column == 'po_returns_mean_flat':
                label = column
            else:
                label = 'mutate_' + str(i)
                i += 1
            plt.plot(df2['iteration'], df2[column], label=label)
    plt.xlabel('iteration')
    plt.ylabel('returns_mean')
    plt.title('Returns of Ant-v3')
    plt.legend(loc='best')
    plt.show()