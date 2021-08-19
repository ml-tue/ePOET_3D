from typing import Any, Sequence, NamedTuple
import numpy as np
import bsuite
from bsuite.baselines import base
from bsuite.baselines import experiment
import dm_env
import optax
import rlax
import haiku as hk
import jax
from BootstrappedDqn import BootstrappedDqn
from reversedgymwrapper import ReverseGymWrapper
import gym
import mujoco_py

class TrainingState(NamedTuple):
  params: hk.Params
  target_params: hk.Params
  opt_state: Any
  step: int

class TDU(BootstrappedDqn):
    def __init__(self, K: int, beta: float,**kwargs: Any):
        """TDU under Bootstrapped DQN with randomized prior functions."""
        super(TDU, self).__init__(**kwargs)
        network, optimizer, N = kwargs['network'], kwargs['optimizer'], kwargs['num_ensemble']
        noise_scale, discount = kwargs['noise_scale'], kwargs['discount']

        # Transform the (impure) network into a pure function.
        network = hk.without_apply_rng(hk.transform(network, apply_rng=True))

        def td(params: hk.Params, target_params: hk.Params,
               transitions: Sequence[jax.numpy.ndarray]) -> jax.numpy.ndarray:
            """TD-error with added reward noise + half-in bootstrap."""
            o_tm1, a_tm1, r_t, d_t, o_t, z_t = transitions
            q_tm1 = network.apply(params, o_tm1)
            q_t = network.apply(target_params, o_t)
            r_t += noise_scale*z_t
            return jax.vmap(rlax.q_learning)(q_tm1, a_tm1, r_t, discount*d_t, q_t)

        def loss(params: Sequence[hk.Params], target_params: Sequence[hk.Params],
                 transitions: Sequence[jax.numpy.ndarray]) -> jax.numpy.ndarray:
            """Q-learning loss with TDU."""
            # Compute TD-errors for first K members.
            o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions
            td_K = [td(params[k], target_params[k],
                       [o_tm1, a_tm1, r_t, d_t, o_t, z_t[:, k]]) for k in range(K)]

            # TDU signal on first K TD-errors.
            r_t += beta*jax.lax.stop_gradient(jax.numpy.std(jax.numpy.stack(td_K, axis=0), axis=0))

            # Compute TD-errors on augmented reward for last K members.
            td_N = [td(params[k], target_params[k],
                       [o_tm1, a_tm1, r_t, d_t, o_t, z_t[:, k]]) for k in range(K, N)]

            return jax.numpy.mean(m_t.T*jax.numpy.stack(td_K + td_N)**2)

        def update(state: TrainingState, gradient: Sequence[jax.numpy.ndarray]) -> TrainingState:
            """Gradient update on ensemble member."""
            updates, new_opt_state = optimizer.update(gradient, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)
            return TrainingState(params=new_params, target_params=state.target_params,
                                 opt_state=new_opt_state, step=state.step + 1)

        @jax.jit
        def sgd_step(states: Sequence[TrainingState],
                     transitions: Sequence[jax.numpy.ndarray]) -> Sequence[TrainingState]:
            """Does a step of SGD for the whole ensemble over ‘transitions‘."""
            params,target_params = zip(*[(state.params, state.target_params) for state in states])
            gradients = jax.grad(loss)(params, target_params, transitions)
            print(params)
            print(target_params)
            return [update(state, gradient) for state, gradient in zip(states, gradients)]

        self._sgd_step = sgd_step  # patch BootDQN sgd_step with TDU sgd_step.

    def update(self, timestep: dm_env.TimeStep, action: base.Action,
               new_timestep: dm_env.TimeStep):
        """Update the agent: add transition to replay and periodically do SGD."""
        if new_timestep.last():
            self._active_head = self._ensemble[np.random.randint(0, self._num_ensemble)]

        mask = np.random.binomial(1, self._mask_prob, self._num_ensemble)
        noise = np.random.randn(self._num_ensemble)
        transition=[timestep.observation,action,np.float32(new_timestep.reward),
                    np.float32(new_timestep.discount),new_timestep.observation,mask,noise]
        self._replay.add(transition)
        if self._replay.size < self._min_replay_size:
            return

        if self._total_steps % self._sgd_period == 0:
            transitions = self._replay.sample(self._batch_size)
            self._ensemble = self._sgd_step(self._ensemble, transitions)

        for k, state in enumerate(self._ensemble):
            if state.step % self._target_update_period == 0:
                self._ensemble[k] = state._replace(target_params=state.params)

from bsuite.logging import csv_logging
from examples.hexapod_trossen_terrain_all import Hexapod
# env = NormalizedActions(gym.make(args.env_name))
# env = gym.make(args.env_name)
# gym_env = Hexapod()
results_dir = "./log/hexapod"
num_episodes = 500
# gym_env = gym.make('Ant-v3')
gym_env = Hexapod()
env = ReverseGymWrapper(gym_env, num_episodes=num_episodes)
prior_scale = 5.
hidden_sizes = [64, 64]
# hidden_sizes = [256, 256]

# env = bsuite.load_and_record_to_csv('catch/0', results_dir=results_dir)
env = csv_logging.wrap_environment(
      env=env,
      bsuite_id='Hexapod/1',
      results_dir=results_dir,
      overwrite=True,
      log_by_step=False
  )

def network(inputs: jax.numpy.ndarray) -> jax.numpy.ndarray:
    """Simple Q-network with randomized prior function."""
    net = hk.nets.MLP([*hidden_sizes, gym_env.action_space.shape[0]])
    prior_net = hk.nets.MLP([*hidden_sizes, gym_env.action_space.shape[0]])
    # net = hk.nets.MLP([*hidden_sizes, 8])
    # prior_net = hk.nets.MLP([*hidden_sizes, 8])
    # net = hk.nets.MLP([*hidden_sizes, env.action_spec().num_values])
    # prior_net = hk.nets.MLP([*hidden_sizes, env.action_spec().num_values])
    x = hk.Flatten()(inputs)
    return net(x) + prior_scale * jax.lax.stop_gradient(prior_net(x))

optimizer = optax.adam(learning_rate=1e-3)
from gym.spaces import Box

agent_tdu = TDU(K=10, beta=0.01,
            obs_spec=env.observation_spec(),#Box(-np.inf, np.inf, (53,), np.float64),  #
                action_spec=env.action_spec(),#Box(-1.0, 1.0, (18,), np.float32), #
              network=network,
              optimizer=optimizer,
              num_ensemble=20,
              batch_size=256, #128,
              discount=.99,
              replay_capacity=1000000, #10000
              min_replay_size=128,
              sgd_period=1,
              target_update_period=1000, #1,
              # target_update_period=4,
              mask_prob=1.0,
              noise_scale=0.02,)

# agent_BootstrappedDqn = BootstrappedDqn(
#             obs_spec=env.observation_spec(),#Box(-np.inf, np.inf, (53,), np.float64),  #
#             action_spec=env.action_spec(),#Box(-1.0, 1.0, (18,), np.float32), #
#               network=network,
#               optimizer=optimizer,
#               num_ensemble=2,
#               batch_size=128,
#               discount=.99,
#               replay_capacity=100000,
#               min_replay_size=128,
#               sgd_period=1,
#               target_update_period=4,
#               mask_prob=1.0,
#               noise_scale=0.,)

experiment.run(
    agent=agent_tdu,
    environment=env,
    num_episodes=num_episodes,
    verbose=False)


# from bsuite.experiments import summary_analysis
# from bsuite.experiments.catch import analysis as catch_analysis
# from bsuite.logging import csv_load
# from bsuite.utils.gym_wrapper import DMEnvFromGym
# from bsuite.baselines.jax.boot_dqn import BootstrappedDqn

# DF, SWEEP_VARS = csv_load.load_bsuite(results_dir)
#
# #@title overall score as radar plot (double-click to show/hide code)
# BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)
# BSUITE_SUMMARY = summary_analysis.ave_score_by_tag(BSUITE_SCORE, SWEEP_VARS)
#
# #@title plotting overall score as bar (double-click to show/hide code)
# summary_analysis.bsuite_bar_plot(BSUITE_SCORE, SWEEP_VARS).draw(show=True)
#
# #@title parsing data
# catch_df = DF[DF.bsuite_env == 'humanoid'].copy()
# summary_analysis.plot_single_experiment(BSUITE_SCORE, 'humanoid', SWEEP_VARS).draw(show=True)
#
# #@title plot average regret through learning (lower is better)
# catch_analysis.plot_learning(catch_df, SWEEP_VARS).draw(show=True)
#
# #@title plot performance by seed (higher is better)
# catch_analysis.plot_seeds(catch_df, SWEEP_VARS).draw()

# def loss(transitions, Q_params, Qtilde_distribution_params, beta):
#     # Estimate TD-error distribution.
#     td_K = array([td_error(p, transitions) for p in sample(Qtilde_distribution_params)])
#     # Compute exploration bonus and Q-function reward.
#     transitions.reward_t += beta*stop_gradient(std(td_K, axis=1))
#     td_N = td_error(Q_params, transitions)
#     # Combine for overall TD-loss.
#     td_errors = concatenate((td_ex, td_in), axis=1)
#     td_loss = mean(0.5*(td_errors)**2))
#     return td_loss
#



# agent_tdu = TDU(K=10, beta=0.01,
#             obs_spec=env.observation_space,
#                 action_spec=env.action_space,
#               network=network,
#               optimizer=optimizer,
#               num_ensemble=10,
#               batch_size=128,
#               discount=.99,
#               replay_capacity=10000,
#               min_replay_size=128,
#               sgd_period=1,
#               target_update_period=4,
#               mask_prob=1.0,
#               noise_scale=0.,)
#
# for _ in range(num_episodes):
#     # Run an episode.
#     timestep = env.reset()
#     while not timestep.last():
#         # Generate an action from the agent's policy.
#         action = agent_tdu.select_action(timestep)
#
#         # Step the environment.
#         new_timestep = env.step(action)
#
#         # Tell the agent about what just happened.
#         agent_tdu.update(timestep, action, new_timestep)
#
#         # Book-keeping.
#         timestep = new_timestep