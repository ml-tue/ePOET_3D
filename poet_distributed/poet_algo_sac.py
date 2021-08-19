# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .logger import CSVLogger
import logging
logger = logging.getLogger(__name__)
import numpy as np
from poet_distributed.es import ESOptimizer
from poet_distributed.es import initialize_worker_fiber
from collections import OrderedDict
from poet_distributed.niches.box2d.env import Env_config
from poet_distributed.niches.box2d.cppn import CppnEnvParams
from poet_distributed.reproduce_ops import Reproducer
from poet_distributed.novelty import compute_novelty_vs_archive
import json
from poet_distributed.niches.box2d.global_replay_buffer import replay_buffer

from gym.spaces import Box
import torch
from torch import optim
from poet_distributed.niches.box2d.model import simulate
import random
import math

from torchrl.replay_buffers import BaseReplayBuffer
from torchrl.utils import Logger
import torchrl.policies as policies
import torchrl.networks as networks
from torchrl.algo import TwinSACQ
from torchrl.collector.base import VecCollector
from torchrl.env import get_vec_env
import os
import os.path as osp
import time
from sklearn.manifold import TSNE

def gen_sac():
    sac_params = {
        "env_name" : "Hexapod",
        "env":{
            "reward_scale":1,
            "obs_norm": False
        },
        "replay_buffer":{
            "size": 1e6,
            "time_limit_filter": False
        },
        "net":{
            "hidden_shapes": [256,256],
            "append_hidden_shapes":[]
        },
        "policy":{
            "tanh_action": True
        },
        "collector":{
            "epoch_frames": 2000,
            "max_episode_frames": 2000,
            "eval_episodes": 1
        },
        "general_setting": {
            "discount" : 0.99,
            "pretrain_epochs" : 1,
            "num_epochs" : 1000,
            "batch_size" : 256,
            "target_hard_update_period" : 1000,
            "use_soft_update" : True,
            "tau" : 0.005,
            "opt_times" : 1000
        },
        "twin_sac_q":{
            "plr" : 3e-4,
            "qlr" : 3e-4,
            "policy_std_reg_weight": 0,
            "policy_mean_reg_weight": 0,
            "reparameterization": True,
            "automatic_entropy_tuning": True
        }
    }
    vec_env_nums = 1
    device = torch.device(
        "cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    env = get_vec_env(
        sac_params["env_name"],
        sac_params["env"],
        vec_env_nums
    )
    eval_env = get_vec_env(
        sac_params["env_name"],
        sac_params["env"],
        vec_env_nums
    )
    if hasattr(env, "_obs_normalizer"):
        eval_env._obs_normalizer = env._obs_normalizer

    env.seed(110)
    torch.manual_seed(110)
    np.random.seed(110)
    random.seed(110)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(110)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    buffer_param = sac_params['replay_buffer']

    experiment_name = 'twin_sac_q_hex'
    logger = Logger(
        experiment_name, sac_params['env_name'],
        110, sac_params, './log', True)

    sac_params['general_setting']['env'] = env
    global replay_buffer
    #replay_buffer = BaseReplayBuffer(
    #    env_nums=vec_env_nums,
    #    max_replay_buffer_size=int(buffer_param['size']),
    #    time_limit_filter=buffer_param['time_limit_filter']
    #)
    sac_params['general_setting']['replay_buffer'] = replay_buffer
    sac_params['general_setting']['logger'] = logger
    sac_params['general_setting']['device'] = device
    sac_params['net']['base_type'] = networks.MLPBase
    sac_params['net']['activation_func'] = torch.nn.ReLU

    pf = policies.GuassianContPolicy(
        input_shape=env.observation_space.shape[0],
        output_shape=2 * env.action_space.shape[0],
        **sac_params['net'],
        **sac_params['policy'])
    qf1 = networks.QNet(
        input_shape=env.observation_space.shape[0] + env.action_space.shape[0],
        output_shape=1,
        **sac_params['net'])
    qf2 = networks.QNet(
        input_shape=env.observation_space.shape[0] + env.action_space.shape[0],
        output_shape=1,
        **sac_params['net'])

    sac_params['general_setting']['collector'] = VecCollector(
        env=env, pf=pf, eval_env=eval_env,
        replay_buffer=replay_buffer, device=device,
        train_render=False,
        **sac_params["collector"]
    )
    sac_params['general_setting']['save_dir'] = osp.join(
        logger.work_dir, "model")
    agent = TwinSACQ(
            pf=pf,
            qf1=qf1,
            qf2=qf2,
            **sac_params["twin_sac_q"],
            **sac_params["general_setting"]
        )
    # twin_sac_model = '/home/fang/project/thesis/torchrl/log/twin_sac_q_hex/Hexapod/123456/model/model_pf_finish.pth'
    pretrained_sac_pf = '/home/TUE/20191160/ant_poet/sac_model/model_pf_best.pth'
    pretrained_sac_qf1 = '/home/TUE/20191160/ant_poet/sac_model/model_qf1_best.pth'
    pretrained_sac_qf2 = '/home/TUE/20191160/ant_poet/sac_model/model_qf2_best.pth'
    agent.pf.load_state_dict(torch.load(pretrained_sac_pf))
    agent.qf1.load_state_dict(torch.load(pretrained_sac_qf1))
    agent.qf2.load_state_dict(torch.load(pretrained_sac_qf2))
    #agent.pf.load_state_dict(torch.load(pretrained_sac_pf, map_location=torch.device('cpu')))
    #agent.qf1.load_state_dict(torch.load(pretrained_sac_qf1, map_location=torch.device('cpu')))
    #agent.qf2.load_state_dict(torch.load(pretrained_sac_qf2, map_location=torch.device('cpu')))
    # agent.train()
    return agent

def construct_niche_fns_from_env(args, env, env_params, seed):
    def niche_wrapper(configs, env_params, seed):  # force python to make a new lexical scope
        def make_niche():
            from poet_distributed.niches import Box2DNiche
            return Box2DNiche(env_configs=configs,
                            env_params=env_params,
                            seed=seed,
                            init=args.init,
                            algs=args.algs,
                            stochastic=args.stochastic)

        return make_niche

    niche_name = env.name
    configs = (env,)

    return niche_name, niche_wrapper(list(configs), env_params, seed)


class MultiESOptimizer:
    def __init__(self, args):

        self.args = args

        import fiber as mp

        mp_ctx = mp.get_context('spawn')
        manager = mp_ctx.Manager()
        self.manager = manager
        self.fiber_shared = {
                "niches": manager.dict(),
                "thetas": manager.dict(),
        }
        self.fiber_pool = mp_ctx.Pool(args.num_workers, initializer=initialize_worker_fiber,
                initargs=(self.fiber_shared["thetas"],
                    self.fiber_shared["niches"]))

        self.ANNECS = 0
        self.env_registry = OrderedDict()
        self.env_archive = OrderedDict()
        self.env_reproducer = Reproducer(args)
        self.optimizers = OrderedDict()
        self.archived_optimizers = OrderedDict()

        self.agent = gen_sac()
        self.sac_episodes = 1
       
        env = Env_config(
            name='flat',
            ground_roughness=0,
            pit_gap=[],
            stump_width=[],
            stump_height=[],
            stump_float=[],
            stair_height=[],
            stair_width=[],
            stair_steps=[])

        params = CppnEnvParams()
        params.save_genome('/home/TUE/20191160/ant_poet/rs1/cppngenomes_admitted/', env.name)
        ##params = CppnEnvParams(genome_path='/home/TUE/20191160/ant_poet/genome_1624798956.482656_saved.pickle')
        ###params = CppnEnvParams(genome_path='/home/TUE/20191160/ant_poet/genome_1624798972.7557998_saved.pickle')
        #params= CppnEnvParams(genome_path='/home/TUE/20191160/ant_poet/rs1/cppngenomes_admitted/14a244e9-94b4-4f10-8507-5666f4768a13.pickle')

        # ##init_model_file = '/home/TUE/20191160/ant_poet/poet_hexA8.e3653398-afec-4958-ac16-5d867338ab64.best.json'
        # ##init_model_file = '/home/TUE/20191160/ant_poet/poet_hexA11.7fc55c4e-3c25-40df-943a-e0be4de8c691.best.json'
        # init_model_file = '/home/TUE/20191160/ant_poet/sac_model/poet_twin22.sac_178.best.json'
        # with open(init_model_file) as f:
        #     tmp_data = json.load(f)
        # print('loaded file %s' % (init_model_file))
        # model_params = tmp_data[0]

        self.add_optimizer(env=env, cppn_params=params, seed=args.master_seed, model_params=None)

    def create_optimizer(self, env, cppn_params, seed, created_at=0, model_params=None, is_candidate=False):

        assert env != None
        assert cppn_params != None

        optim_id, niche_fn = construct_niche_fns_from_env(args=self.args, env=env, env_params=cppn_params, seed=seed)

        niche = niche_fn()
        if model_params is not None:
            theta = np.array(model_params)
        else:
            theta=niche.initial_theta()
        assert optim_id not in self.optimizers.keys()

        return ESOptimizer(
            optim_id=optim_id,
            fiber_pool=self.fiber_pool,
            fiber_shared=self.fiber_shared,
            theta=theta,
            make_niche=niche_fn,
            learning_rate=self.args.learning_rate,
            lr_decay=self.args.lr_decay,
            lr_limit=self.args.lr_limit,
            batches_per_chunk=self.args.batches_per_chunk,
            batch_size=self.args.batch_size,
            eval_batch_size=self.args.eval_batch_size,
            eval_batches_per_step=self.args.eval_batches_per_step,
            l2_coeff=self.args.l2_coeff,
            noise_std=self.args.noise_std,
            noise_decay=self.args.noise_decay,
            normalize_grads_by_noise_std=self.args.normalize_grads_by_noise_std,
            returns_normalization=self.args.returns_normalization,
            noise_limit=self.args.noise_limit,
            log_file=self.args.log_file,
            created_at=created_at,
            is_candidate=is_candidate)


    def add_optimizer(self, env, cppn_params, seed, created_at=0, model_params=None):
        '''
            creat a new optimizer/niche
            created_at: the iteration when this niche is created
        '''
        o = self.create_optimizer(env, cppn_params, seed, created_at, model_params)
        optim_id = o.optim_id
        self.optimizers[optim_id] = o

        assert optim_id not in self.env_registry.keys()
        assert optim_id not in self.env_archive.keys()
        self.env_registry[optim_id] = (env, cppn_params)
        self.env_archive[optim_id] = (env, cppn_params)

    def archive_optimizer(self, optim_id):
        assert optim_id in self.optimizers.keys()
        #assume optim_id == env_id for single_env niches
        o = self.optimizers.pop(optim_id)
        assert optim_id in self.env_registry.keys()
        self.env_registry.pop(optim_id)
        logger.info('Archived {} '.format(optim_id))
        self.archived_optimizers[optim_id] = o

    def ind_es_step(self, iteration):
        tasks = [o.start_step() for o in self.optimizers.values()]

        for optimizer, task in zip(self.optimizers.values(), tasks):

            optimizer.theta, stats = optimizer.get_step(task)
            self_eval_task = optimizer.start_theta_eval(optimizer.theta)
            self_eval_stats = optimizer.get_theta_eval(self_eval_task)

            logger.info('Iter={} Optimizer {} theta_mean {} best po {} iteration spent {}'.format(
                iteration, optimizer.optim_id, self_eval_stats.eval_returns_mean,
                stats.po_returns_max, iteration - optimizer.created_at))

            optimizer.update_dicts_after_es(stats=stats,
                self_eval_stats=self_eval_stats)

    def transfer(self, propose_with_adam, checkpointing, reset_optimizer):
        logger.info('Computing direct transfers...')
        proposal_targets = {}
        for source_optim in self.optimizers.values():
            source_tasks = []
            proposal_targets[source_optim] = []
            for target_optim in [o for o in self.optimizers.values()
                                    if o is not source_optim]:
                task = target_optim.start_theta_eval(
                    source_optim.theta)
                source_tasks.append((task, target_optim))

            for task, target_optim in source_tasks:
                stats = target_optim.get_theta_eval(task)

                try_proposal = target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                                    source_optim_theta=source_optim.theta,
                                    stats=stats, keyword='theta')
                if try_proposal:
                    proposal_targets[source_optim].append(target_optim)

        logger.info('Computing proposal transfers...')
        for source_optim in self.optimizers.values():
            source_tasks = []
            for target_optim in [o for o in self.optimizers.values()
                                    if o is not source_optim]:
                if target_optim in proposal_targets[source_optim]:
                    task = target_optim.start_step(source_optim.theta)
                    source_tasks.append((task, target_optim))

            for task, target_optim in source_tasks:
                proposed_theta, _ = target_optim.get_step(
                    task, propose_with_adam=propose_with_adam, propose_only=True)

                proposal_eval_task = target_optim.start_theta_eval(proposed_theta)
                proposal_eval_stats = target_optim.get_theta_eval(proposal_eval_task)

                target_optim.update_dicts_after_transfer(source_optim_id=source_optim.optim_id,
                    source_optim_theta=proposed_theta,
                    stats=proposal_eval_stats, keyword='proposal')

        logger.info('Considering transfers...')
        for o in self.optimizers.values():
            o.pick_proposal(checkpointing, reset_optimizer)

    def check_optimizer_status(self, iteration):
        '''
            return two lists
        '''
        logger.info("health_check")
        repro_candidates, delete_candidates = [], []
        for optim_id in self.env_registry.keys():
            o = self.optimizers[optim_id]
            logger.info("niche {} created at {} start_score {} current_self_evals {}".format(
                optim_id, o.created_at, o.start_score, o.self_evals))
            if o.self_evals >= self.args.repro_threshold:
                repro_candidates.append(optim_id)

        logger.debug("candidates to reproduce")
        logger.debug(repro_candidates)
        logger.debug("candidates to delete")
        logger.debug(delete_candidates)

        return repro_candidates, delete_candidates


    def pass_dedup(self, env_config):
        if env_config.name in self.env_registry.keys():
            logger.debug("active env already. reject!")
            return False
        else:
            return True

    def pass_mc(self, score):
        if score < self.args.mc_lower or score > self.args.mc_upper:
            return False
        else:
            return True

    def get_new_env(self, list_repro):
        print('get_new_env starts')
        optim_id = self.env_reproducer.pick(list_repro)
        assert optim_id in self.optimizers.keys()
        assert optim_id in self.env_registry.keys()
        parent_env_config, parent_cppn_params = self.env_registry[optim_id]
        child_env_config = self.env_reproducer.mutate(parent_env_config, no_mutate=True)
        child_cppn_params = parent_cppn_params.get_mutated_params()

        logger.info("we pick to mutate: {} and we got {} back".format(optim_id, child_env_config.name))
        logger.debug("parent")
        logger.debug(parent_env_config)
        logger.debug("child")
        logger.debug(child_env_config)

        seed = np.random.randint(1000000)
        return child_env_config, child_cppn_params, seed, optim_id

    def get_child_list(self, parent_list, max_children):
        print('get_child_list starts')
        child_list = []

        mutation_trial = 0
        while mutation_trial < max_children:
            new_env_config, new_cppn_params, seed, parent_optim_id = self.get_new_env(parent_list)
            mutation_trial += 1
            if self.pass_dedup(new_env_config):
                o = self.create_optimizer(new_env_config, new_cppn_params, seed, is_candidate=True)
                score = o.evaluate_theta(self.optimizers[parent_optim_id].theta)
                if self.pass_mc(score):
                    novelty_score = compute_novelty_vs_archive(self.archived_optimizers, self.optimizers, o, k=5,
                                        low=self.args.mc_lower, high=self.args.mc_upper)
                    logger.debug("{} passed mc, novelty score {}".format(score, novelty_score))
                    child_list.append((new_env_config, new_cppn_params, seed, parent_optim_id, novelty_score))
                del o

        #sort child list according to novelty for high to low
        child_list = sorted(child_list,key=lambda x: x[4], reverse=True)
        return child_list

    def adjust_envs_niches(self, iteration, steps_before_adjust, max_num_envs=None, max_children=8, max_admitted=1):

        if iteration > 0 and iteration % steps_before_adjust == 0:
            list_repro, list_delete = self.check_optimizer_status(iteration)

            if len(list_repro) == 0:
                return

            logger.info("list of niches to reproduce")
            logger.info(list_repro)
            logger.info("list of niches to delete")
            logger.info(list_delete)

            for optim in self.optimizers.values():
                optim.update_pata_ec(self.archived_optimizers, self.optimizers, self.args.mc_lower, self.args.mc_upper)

            for optim in self.archived_optimizers.values():
                optim.update_pata_ec(self.archived_optimizers, self.optimizers, self.args.mc_lower, self.args.mc_upper)

            child_list = self.get_child_list(list_repro, max_children)

            if child_list == None or len(child_list) == 0:
                logger.info("mutation to reproduce env FAILED!!!")
                return
            #print(child_list)
            admitted = 0
            for child in child_list:
                new_env_config, new_cppn_params, seed, _, _ = child
                # targeted transfer
                o = self.create_optimizer(new_env_config, new_cppn_params, seed, is_candidate=True)
                score_child, theta_child = o.evaluate_transfer(self.optimizers)
                if self.archived_optimizers != None:
                    score_archive, _ = o.evaluate_transfer(self.archived_optimizers, evaluate_proposal=False)
                del o
                if self.pass_mc(score_child):  # check mc
                    self.add_optimizer(env=new_env_config, cppn_params=new_cppn_params, seed=seed, created_at=iteration, model_params=np.array(theta_child))
                    print('admitted env: ', new_cppn_params)
                    new_cppn_params.save_genome('/home/TUE/20191160/ant_poet/rs1/cppngenomes_admitted/', new_env_config.name)
                    admitted += 1
                    if score_archive !=None:
                        if self.pass_mc(score_archive):
                            self.ANNECS += 1
                    if admitted >= max_admitted:
                        break

            if max_num_envs and len(self.optimizers) > max_num_envs:
                num_removals = len(self.optimizers) - max_num_envs
                self.remove_oldest(num_removals)

    def remove_oldest(self, num_removals):
        list_delete = []
        for optim_id in self.env_registry.keys():
            if len(list_delete) < num_removals:
                list_delete.append(optim_id)
            else:
                break

        for optim_id in list_delete:
            self.archive_optimizer(optim_id)           

    def optimize(self, iterations=200,
                 steps_before_transfer=25,
                 propose_with_adam=False,
                 checkpointing=False,
                 reset_optimizer=True):

        total_numsteps = 0

        for iteration in range(iterations):

            if iteration < 10:
                self.sac_episodes = 10
            else:
                self.sac_episodes = 1

            # ##crossover between es optimizers
            #if len(self.optimizers.values()) >= 2 and iteration % (steps_before_transfer*4) == (steps_before_transfer*2-1) and iteration > 0:
             #   self.inner_crossover(iteration)

            if len(self.optimizers.values()) >= 2 and iteration % (steps_before_transfer*4) == (steps_before_transfer*4-1) and iteration > 0:
                self.inner_crossover(iteration)

            ## SAC update
            #if iteration % (steps_before_transfer*2) == (steps_before_transfer*2-5) and iteration > 0:
                #self.copy_single_sac(iteration)
             #   self.crossover(iteration)

            if iteration % (steps_before_transfer*4) == (steps_before_transfer*4-2) and iteration > 0:
                self.copy_single_sac(iteration)

            if iteration > self.args.adjust_interval * steps_before_transfer:
                self.adjust_envs_niches(iteration, steps_before_transfer,
                                        max_num_envs=self.args.max_num_envs, max_admitted=1)
            else:
                self.adjust_envs_niches(iteration, self.args.adjust_interval * steps_before_transfer,
                                        max_num_envs=self.args.max_num_envs, max_admitted=1)

            for o in self.optimizers.values():
                o.clean_dicts_before_iter()

            self.ind_es_step(iteration=iteration)

            total_numsteps += 2000
            self.train_sac(total_numsteps, iteration)

            if len(self.optimizers) > 1 and iteration % steps_before_transfer == 0:
                self.transfer(propose_with_adam=propose_with_adam,
                              checkpointing=checkpointing,
                              reset_optimizer=reset_optimizer)

                # ## choose the best theta to update sac actor after transfer
                #self.copy_to_sac()
                global replay_buffer
                print('replay buffer is updated')
                self.agent.replay_buffer = replay_buffer
                # # for sample_dict in replay_buffer:
                # #     self.agent.replay_buffer.add_sample(sample_dict)

            if iteration % steps_before_transfer == 0:
                for o in self.optimizers.values():
                    o.save_to_logger(iteration)


    def train_sac(self, total_numsteps, iteration):
        for epoch in range(self.sac_episodes):
            self.agent.current_epoch = epoch
            start = time.time()
            self.agent.start_epoch()
            explore_start_time = time.time()
            training_epoch_info = self.agent.collector.train_one_epoch()
            for reward in training_epoch_info["train_rewards"]:
                self.agent.training_episode_rewards.append(reward)
            self.agent.explore_time += time.time() - explore_start_time
            train_start_time = time.time()
            self.agent.update_per_epoch()
            self.agent.train_time += time.time() - train_start_time
            finish_epoch_info = self.agent.finish_epoch()
            total_numsteps += self.agent.epoch_frames
            print('sac episode_reward is {} in episode {}'.format(training_epoch_info["train_epoch_reward"], iteration))

            if iteration % 50 == 0:
            # if epoch % self.agent.eval_interval == 0:
                eval_start_time = time.time()
                eval_infos = self.agent.collector.eval_one_epoch()
                eval_time = time.time() - eval_start_time
                infos = {}
                for reward in eval_infos["eval_rewards"]:
                    self.agent.episode_rewards.append(reward)
                if self.agent.best_eval is None or \
                    (np.mean(eval_infos["eval_rewards"]) > self.agent.best_eval):
                    self.agent.best_eval = np.mean(eval_infos["eval_rewards"])
                    self.agent.snapshot(self.agent.save_dir, 'best')
                del eval_infos["eval_rewards"]
                infos["Running_Average_Rewards"] = np.mean(
                    self.agent.episode_rewards)
                infos["Train_Epoch_Reward"] = \
                    training_epoch_info["train_epoch_reward"]
                infos["Running_Training_Average_Rewards"] = np.mean(
                    self.agent.training_episode_rewards)
                infos["Explore_Time"] = self.agent.explore_time
                infos["Train___Time"] = self.agent.train_time
                infos["Eval____Time"] = eval_time
                self.agent.explore_time = 0
                self.agent.train_time = 0
                infos.update(eval_infos)
                infos.update(finish_epoch_info)
                self.agent.logger.add_epoch_info(
                    epoch, total_numsteps, time.time() - start, infos)
                self.agent.start = time.time()
            # # if epoch % self.agent.save_interval == 0:
            # #     self.agent.snapshot(self.agent.save_dir, epoch)
        if iteration % 50 == 0:
            self.agent.snapshot(self.agent.save_dir, iteration)
            self.agent.collector.terminate()

    def copy_single_sac(self, iteration):
        #odict_keys(['base.seq_fcs.0.weight', 'base.seq_fcs.0.bias', 'base.seq_fcs.2.weight', 'base.seq_fcs.2.bias', 'seq_append_fcs.0.weight', 'seq_append_fcs.0.bias'])

        sac_para = self.agent.pf.state_dict()
        theta = []
        ##random reshapping
        for idx, key in enumerate(sac_para.keys()):
            W1 = sac_para[key]
            W1 = W1.cpu().detach().numpy()
            if idx < 4:
                if len(W1.shape) == 2:
                    tmp = W1.T
                    if tmp.shape[0] == 53:
                        tmp_a = np.random.choice(tmp.ravel(), size=53*40).reshape((53,40))
                    else:
                        tmp_a = np.random.choice(tmp.ravel(), size=40*40).reshape((40, 40))
                    #print(tmp_a.shape)
                    theta.append(tmp_a.flatten())
                elif len(W1.shape) == 1:
                    tmp = np.expand_dims(W1, axis=0)
                    tmp_a = np.random.choice(tmp.ravel(), size=1*40).reshape((1, 40))
                    tmp_a = tmp_a[0]
                    #print(tmp_a.shape)
                    theta.append(tmp_a.flatten())
            else:
                if len(W1.shape) == 2:
                    tmp = W1.T
                    if tmp.shape[0] > 40 and tmp.shape[1] < 40:
                        tmp_a = np.random.choice(tmp.ravel(), size=40*18).reshape((40, 18))
                        #print(tmp_a.shape)
                        theta.append(tmp_a.flatten())
                elif len(W1.shape) == 1:
                    tmp = np.expand_dims(W1, axis=0)
                    if tmp.shape[1] > 18 and tmp.shape[1] < 40:
                        tmp_a = np.random.choice(tmp.ravel(), size=1*18).reshape((1, 18))
                    tmp_a = tmp_a[0]
                    #print(tmp_a.shape)
                    theta.append(tmp_a.flatten())

        ##tsne
        #for idx, key in enumerate(sac_para.keys()):
         #   W1 = sac_para[key]
          #  W1 = W1.cpu().detach().numpy()
           # if idx < 4:
            #    if len(W1.shape) == 2:
             #       tmp = W1.T
              #      if tmp.shape[1] > 40:
               #         if tmp.shape[0] == tmp.shape[1]:
                #            tmp_a = TSNE(n_components=40, method='exact').fit_transform(tmp)
                 #           tmp_a = tmp_a.T
                  #          tmp_a = TSNE(n_components=40, method='exact').fit_transform(tmp_a)
                   #         tmp_a = tmp_a.T
                    #    else:
                     #       tmp_a = TSNE(n_components=40, method='exact').fit_transform(tmp)
                    ##print(tmp_a.shape)
                    #theta.append(tmp_a.flatten())
                #elif len(W1.shape) == 1:
                 #   tmp = np.expand_dims(W1, axis=0)
                  #  if tmp.shape[1] > 40:
                   #     tmp_a = TSNE(n_components=40, method='exact').fit_transform(tmp)
                    #tmp_a = tmp_a[0]
                    ##print(tmp_a.shape)
                    #theta.append(tmp_a.flatten())
            #else:
             #   if len(W1.shape) == 2:
              #      tmp = W1.T
               #     if tmp.shape[0] > 40 and tmp.shape[1] < 40:
                #        tmp_a = TSNE(n_components=18, method='exact').fit_transform(tmp)
                 #       tmp_a = tmp_a.T
                  #      tmp_a = TSNE(n_components=40, method='exact').fit_transform(tmp_a)
                   #     tmp_a = tmp_a.T
                    #    #print(tmp_a.shape)
                     #   theta.append(tmp_a.flatten())
                #elif len(W1.shape) == 1:
                 #   tmp = np.expand_dims(W1, axis=0)
                  #  if tmp.shape[1] > 18 and tmp.shape[1] < 40:
                   #     tmp_a = TSNE(n_components=18, method='exact').fit_transform(tmp)
                    #tmp_a = tmp_a[0]
                    ##print(tmp_a.shape)
                    #theta.append(tmp_a.flatten())

        ## copy with same size
        #for idx, key in enumerate(sac_para.keys()):
         #   W1 = sac_para[key]
          #  W1 = W1.cpu().detach().numpy()
           # if idx < 4:
            #    if len(W1.shape) == 2:
             #       theta.append((W1.T).flatten())
              #  elif len(W1.shape) == 1:
               #     theta.append(W1.flatten())
           # else:
            #    if len(W1.shape) == 2:
             #       theta.append(((W1.T)[:, :18]).flatten())
              #  elif len(W1.shape) == 1:
               #     theta.append(W1[:18].flatten())

        print('add sac actor to new theta')
        env = Env_config(
            name='sac_' + str(iteration),
            ground_roughness=0,
            pit_gap=[],
            stump_width=[],
            stump_height=[],
            stump_float=[],
            stair_height=[],
            stair_width=[],
            stair_steps=[])
        params = CppnEnvParams()
        rs_seeds = np.random.randint(np.int32(2 ** 31 - 1), size=256)
        random_state = np.random.RandomState(rs_seeds).randint(1000000)
        self.add_optimizer(env, params, random_state, created_at=iteration, model_params=np.hstack(theta))
        params.save_genome('/home/TUE/20191160/ant_poet/rs1/cppngenomes_admitted/', env.name)

    def copy_to_sac(self):
        niches = self.fiber_shared["niches"]
        thetas = self.fiber_shared['thetas']
        #max_return = 0
        for i, optim in enumerate(self.optimizers.values()):
            niche = niches[optim.optim_id]
            theta = thetas[optim.optim_id]
            rs_seeds = np.random.randint(np.int32(2 ** 31 - 1), size=256)
            random_state = np.random.RandomState(rs_seeds)
            old_return, _ = niche.rollout(theta, random_state, eval=True)
            if i == 0:
                max_return = old_return
                best_niche = niche
            if i>0 and old_return > max_return:
                max_return = old_return
                best_niche = niche
        ##update new values to sac agent
        for idx, param_tensor in enumerate(self.agent.pf.state_dict()):
            if len(self.agent.pf.state_dict()[param_tensor].size()) == 2:
                if idx < 4:
                    for i in range(self.agent.pf.state_dict()[param_tensor].shape[0]):
                        self.agent.pf.state_dict()[param_tensor][i] = torch.from_numpy((np.array(best_niche.model.weight)[math.ceil(idx / 2)].T)[i,:])
                elif idx == 4:
                    for i in range(int(self.agent.pf.state_dict()[param_tensor].shape[0] / 2)):
                        self.agent.pf.state_dict()[param_tensor][i] = torch.from_numpy((np.array(best_niche.model.weight)[math.ceil(idx / 2)].T)[i,:])
            elif len(self.agent.pf.state_dict()[param_tensor].size()) == 1:
                if idx < 5:
                    self.agent.pf.state_dict()[param_tensor] = torch.from_numpy(
                        (np.array(best_niche.model.bias)[math.floor(idx / 2)]))
                elif idx == 5:
                    self.agent.pf.state_dict()[param_tensor][:int(self.agent.pf.state_dict()[param_tensor].shape[0] / 2)] = torch.from_numpy(
                        (np.array(best_niche.model.bias)[math.floor(idx / 2)]))
        print('SAC actor is updated')


    def inner_crossover(self, iteration):
        niches = self.fiber_shared["niches"]
        thetas = self.fiber_shared['thetas']
        tobe_archived = []
        tobe_added = []
        new_thetas = []
        print('self.optimizers.keys: ',self.optimizers.keys())
        for i, optim1 in enumerate(self.optimizers.values()):
            for j, optim2 in enumerate(self.optimizers.values()):
                if optim1.optim_id != optim2.optim_id and i < j:
                    niche1 = niches[optim1.optim_id]
                    theta1 = thetas[optim1.optim_id]
                    niche2 = niches[optim2.optim_id]
                    theta2 = thetas[optim2.optim_id]

                    rs_seeds = np.random.randint(np.int32(2 ** 31 - 1), size=256)
                    random_state = np.random.RandomState(rs_seeds)
                    old_return1, _ = niche1.rollout(theta1, random_state, eval=True)
                    old_return2, _ = niche2.rollout(theta2, random_state, eval=True)

                    seed = random_state.randint(1000000)
                    new_niche1, new_theta1, new_niche2, new_theta2 = self.inner_crossover_single(niche1, niche2)
                    new_return1, _ = simulate(
                        new_niche1.model, seed=seed, train_mode=False, num_episode=1)
                    new_return2, _ = simulate(
                        new_niche2.model, seed=seed, train_mode=False, num_episode=1)
                    print('new {} return1 is {} and old return1 is {}.'.format(optim1.optim_id, new_return1, old_return1))
                    print('new {} return2 is {} and old return2 is {}.'.format(optim2.optim_id, new_return2, old_return2))
                    if len(new_return1) == 1 and new_return1[0] > old_return1:
                        print('change to new theta1')
                        if optim1.optim_id not in tobe_added:
                            tobe_added.append(optim1.optim_id)
                            new_thetas.append(new_theta1)
                        if optim1.optim_id not in tobe_archived:
                            tobe_archived.append(optim1.optim_id)
                    if len(new_return2) == 1 and new_return2[0] > old_return2:
                        print('change to new theta2')
                        if optim2.optim_id not in tobe_added:
                            tobe_added.append(optim2.optim_id)
                            new_thetas.append(new_theta2)
                        if optim2.optim_id not in tobe_archived:
                            tobe_archived.append(optim2.optim_id)
        print('to be added list: ', tobe_added)
        for optim_id, theta in zip(tobe_added, new_thetas):
            niche = niches[optim_id]
            env = Env_config(
                name=optim_id + '_iter' + str(iteration) + '_' + str(time.time()),
                ground_roughness=0,
                pit_gap=[],
                stump_width=[],
                stump_height=[],
                stump_float=[],
                stair_height=[],
                stair_width=[],
                stair_steps=[])
            self.add_optimizer(env, niche.env_params, niche.seed, created_at=iteration, model_params=theta)
        print('to be archived list: ', tobe_archived)
        for optim_id in tobe_archived:
            self.archive_optimizer(optim_id)

    def inner_crossover_single(self, niche1, niche2):
        import random
        theta1 = []
        theta2 = []
        for i in range(3):
            num_variables = np.array(niche1.model.weight)[i].shape[0]
            num_cross_overs = random.randrange(num_variables * 2)
            for _ in range(num_cross_overs):
                receiver_choice = random.random()  # Choose which gene to receive the perturbation
                ind_cr = random.randrange(num_variables)
                if receiver_choice < 0.5:
                    np.array(niche1.model.weight)[i][ind_cr, :] = np.array(niche2.model.weight)[i][ind_cr, :]
                else:
                    np.array(niche2.model.weight)[i][ind_cr, :] = np.array(niche1.model.weight)[i][ind_cr, :]
            theta1.append(np.array(niche1.model.weight)[i].flatten())
            theta2.append(np.array(niche2.model.weight)[i].flatten())

            num_variables_bias = np.array(niche1.model.bias)[i].shape[0]
            num_cross_overs_bias = random.randrange(num_variables_bias)
            for _ in range(num_cross_overs_bias):
                receiver_choice_bias = random.random()  # Choose which gene to receive the perturbation
                ind_cr = random.randrange(num_variables_bias)
                if receiver_choice_bias < 0.5:
                    np.array(niche1.model.bias)[i][ind_cr] = np.array(niche2.model.bias)[i][ind_cr]
                else:
                    np.array(niche2.model.bias)[i][ind_cr] = np.array(niche1.model.bias)[i][ind_cr]
            theta1.append(np.array(niche1.model.bias)[i].flatten())
            theta2.append(np.array(niche2.model.bias)[i].flatten())

        return niche1, np.hstack(theta1), niche2, np.hstack(theta2)

    def crossover(self, iteration):
        niches = self.fiber_shared["niches"]
        thetas = self.fiber_shared['thetas']
        # sac_para = self.agent.policy.state_dict()
        for optim in self.optimizers.values():
            niche = niches[optim.optim_id]
            theta = thetas[optim.optim_id]

            rs_seeds = np.random.randint(np.int32(2 ** 31 - 1), size=256)
            random_state = np.random.RandomState(rs_seeds)
            old_return, _ = niche.rollout(theta, random_state, eval=True)

            seed = random_state.randint(1000000)
            # niche_new, new_theta = self.crossover_single(niche, sac_para)
            niche_new, new_theta = self.crossover_single_sac(niche)
            new_return, _ = simulate(
                niche_new.model, seed=seed, train_mode=False, num_episode=1)
            print('new return is {} and old return is {}.'.format(new_return, old_return))
            if len(new_return) == 1 and new_return[0] > old_return:
                print('change to new theta')
                env = Env_config(
                    name=optim.optim_id + '_sac_' + str(iteration),
                    ground_roughness=0,
                    pit_gap=[],
                    stump_width=[],
                    stump_height=[],
                    stump_float=[],
                    stair_height=[],
                    stair_width=[],
                    stair_steps=[])
                # print('new return is {} and old return is {}, change to new theta.'.format(new_return, old_return))
                self.add_optimizer(env, niche.env_params, niche.seed, created_at=iteration, model_params=new_theta)
                self.archive_optimizer(optim.optim_id)

    def crossover_single_sac(self, niche):
        import math, random
        #odict_keys(['base.seq_fcs.0.weight', 'base.seq_fcs.0.bias', 'base.seq_fcs.2.weight', 'base.seq_fcs.2.bias', 'seq_append_fcs.0.weight', 'seq_append_fcs.0.bias'])

        sac_para = self.agent.pf.state_dict()
        theta = []
        for idx, key in enumerate(sac_para.keys()):
            # References to the variable tensors
            W1 = sac_para[key]
            W1 = W1.cpu().detach().numpy()
            if idx < 6:
                if len(W1.shape) == 2:  # Weights no bias
                    W2 = np.array(niche.model.weight)[math.ceil(idx / 2)].T
                    # print('W1 shape is {}, W2 shape is {} in idx {}'.format(W1.shape, W2.shape, idx))
                    num_variables = W2.shape[0]
                    # Crossover opertation [Indexed by row]
                    num_cross_overs = random.randrange(num_variables * 2)  # Lower bounded on full swaps
                    for i in range(num_cross_overs):
                        receiver_choice = random.random()  # Choose which gene to receive the perturbation
                        if receiver_choice < 0.5:
                            ind_cr = random.randrange(num_variables)  #
                            if W1.shape[1] == W2.shape[1]:
                                W1[ind_cr, :] = W2[ind_cr, :]
                            else:
                                W1[ind_cr, ind_cr:ind_cr+W2.shape[1]] = W2[ind_cr, :]
                        else:
                            ind_cr = random.randrange(num_variables)  #
                            ind_cr1 = random.randrange(W1.shape[0])
                            if W1.shape[1] == W2.shape[1]:
                                W2[ind_cr, :] = W1[ind_cr1, :]
                            else:
                                W2[ind_cr, :] = W1[ind_cr1, ind_cr:ind_cr + W2.shape[1]]
                    niche.model.weight[math.ceil(idx / 2)] = W2.T
                    theta.append((W2.T).flatten())

                elif len(W1.shape) == 1:  # Bias
                    W2 = np.array(niche.model.bias)[math.floor(idx / 2)]
                    # print('W1 shape is {}, W2 bias shape is {} in i {}'.format(W1.shape, W2.shape, idx))
                    num_variables = W2.shape[0]
                    # Crossover opertation [Indexed by row]
                    num_cross_overs = random.randrange(num_variables)  # Lower bounded on full swaps
                    for i in range(num_cross_overs):
                        receiver_choice = random.random()  # Choose which gene to receive the perturbation
                        if receiver_choice < 0.5:
                            ind_cr = random.randrange(num_variables)  #
                            W1[ind_cr] = W2[ind_cr]
                        else:
                            ind_cr = random.randrange(W1.shape[0])  #
                            ind_cr2 = random.randrange(W2.shape[0])  #
                            W2[ind_cr2] = W1[ind_cr]
                    np.array(niche.model.bias)[math.floor(idx / 2)] = W2
                    theta.append(W2.flatten())
            else:
                break

        return niche, np.hstack(theta)
