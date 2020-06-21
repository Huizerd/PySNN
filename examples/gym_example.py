import gym
import torch

from pysnn.network import SpikingModule
from pysnn.neuron import AdaptiveLIFNeuron, LIFNeuron
from pysnn.connection import Linear
from pysnn.learning import MSTDPET


# Do we want hyperparameter optimization?
# TODO: not sure this works correctly
optimization = False
if optimization:
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.suggest.bayesopt import BayesOptSearch


########################################################
# Network
########################################################
class Network(SpikingModule):
    def __init__(self, n_in_dynamics, n_hid_dynamics, n_out_dynamics, c_dynamics):
        super(Network, self).__init__()

        # Input layer
        # 4-dimensional observation of + and -, so 8 neurons
        self.neuron0 = AdaptiveLIFNeuron((1, 1, 8), *n_in_dynamics)

        # Hidden layer
        # Adaptive neuron to cope with highly varying input
        self.neuron1 = AdaptiveLIFNeuron((1, 1, 64), *n_hid_dynamics)

        # Output layer
        # Non-adaptive neuron to decrease latency in action selection
        # Action is binary, so single neuron suffices
        self.neuron2 = LIFNeuron((1, 1, 1), *n_out_dynamics)

        # Connections
        # Weights initialized uniformly in [0, 1]
        self.linear1 = Linear(8, 64, *c_dynamics)
        self.linear2 = Linear(64, 1, *c_dynamics)

    def forward(self, input):
        # Input
        x, t = self.neuron0(input)

        # Hidden layer
        # Connection trace (2nd argument) is not used
        x, t = self.linear1(x, t)
        x, t = self.neuron1(x)

        # Output layer
        x, t = self.linear2(x, t)
        x, t = self.neuron2(x)

        return x

    def encode(self, x):
        # Repeat the input
        x = x.repeat(1, 1, 2)

        # Clamp first half to positive, second to negative
        x[..., :4].clamp_(min=0)
        x[..., 4:].clamp_(max=0)

        # Make absolute
        return x.abs().float()

    def decode(self, x):
        return x.byte()

    def reset_state(self):
        for module in self.spiking_children():
            module.reset_state()

    def step(self, obs, env, rule, render=False):
        # Encode observation
        obs = torch.from_numpy(obs).view(1, 1, -1)
        obs = self.encode(obs)

        # Push through network
        action = self.forward(obs)

        # Decode
        action = self.decode(action)

        # Optional render of environment
        if render:
            env.render()

        # Do environment step
        action = action.item()
        obs, reward, done, _ = env.step(action)

        # Do learning step
        rule.step(reward)

        # Return stepped environment and its returns
        return obs, reward, done, env, rule


########################################################
# Main
########################################################
def main(config):
    # Put config in neuron/connection/rule dicts
    neuron_in = [
        config["thresh0"],
        config["v_rest"],
        config["alpha_v0"],
        config["alpha_t0"],
        config["dt"],
        config["refrac"],
        config["tau_v0"],
        config["tau_t0"],
        config["alpha_thresh0"],
        config["tau_thresh0"],
    ]
    neuron_hid = [
        config["thresh1"],
        config["v_rest"],
        config["alpha_v1"],
        config["alpha_t1"],
        config["dt"],
        config["refrac"],
        config["tau_v1"],
        config["tau_t1"],
        config["alpha_thresh1"],
        config["tau_thresh1"],
    ]
    neuron_out = [
        config["thresh2"],
        config["v_rest"],
        config["alpha_v2"],
        config["alpha_t2"],
        config["dt"],
        config["refrac"],
        config["tau_v2"],
        config["tau_t2"],
    ]
    conns = [config["batch_size"], config["dt"], config["delay"]]
    lr = [config["lr"], config["a_pre"], config["a_post"], config["tau_e_trace"]]

    # Build network
    # Build learning rule from network layers
    net = Network(neuron_in, neuron_hid, neuron_out, conns)
    layers, _, _, _ = net.trace_graph(torch.zeros((1, 1, 8)))
    rule = MSTDPET(layers, *lr)

    # Build env
    env = gym.make("CartPole-v1")
    obs = env.reset()

    # Logging variables
    episode_reward = 0.0

    # Simulation loop
    for step in range(config["steps"]):
        obs, reward, done, env, rule = net.step(obs, env, rule, render=False)
        episode_reward += reward

        # Episode end
        if done:
            obs = env.reset()
            net.reset_state()
            rule.reset_state()
            if optimization:
                tune.track.log(reward=episode_reward)
            episode_reward = 0.0

    # Cleanup
    env.close()


#########################################################
# Training
#########################################################
if __name__ == "__main__":
    # Fixed parameters
    config = {
        "dt": 1.0,
        "thresh0": 0.2,
        "thresh1": 0.2,
        "refrac": 0,
        "v_rest": 0.0,
        "batch_size": 1,
        "delay": 0,
        "a_post": 0.0,
        "steps": 10000,
    }

    if optimization:
        # Search space for Bayesian Optimization
        space = {
            "thresh2": (0.0, 1.0),
            "alpha_v0": (0.0, 2.0),
            "alpha_v1": (0.0, 2.0),
            "alpha_v2": (0.0, 2.0),
            "alpha_t0": (0.0, 2.0),
            "alpha_t1": (0.0, 2.0),
            "alpha_t2": (0.0, 2.0),
            "alpha_thresh0": (0.0, 2.0),
            "alpha_thresh1": (0.0, 2.0),
            "tau_v0": (0.0, 1.0),
            "tau_v1": (0.0, 1.0),
            "tau_v2": (0.0, 1.0),
            "tau_t0": (0.0, 1.0),
            "tau_t1": (0.0, 1.0),
            "tau_t2": (0.0, 1.0),
            "tau_thresh0": (0.0, 1.0),
            "tau_thresh1": (0.0, 1.0),
            "a_pre": (0.0, 2.0),
            "lr": (1e-6, 1e-2),
            "tau_e_trace": (0.0, 1.0),
        }

        # Run hyperparameter search
        search = BayesOptSearch(
            space,
            max_concurrent=6,
            metric="reward",
            mode="max",
            utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
        )
        scheduler = ASHAScheduler(metric="reward", mode="max")
        tune.run(
            main,
            num_samples=100,
            scheduler=scheduler,
            search_alg=search,
            config=config,
            verbose=1,
            local_dir="ray_runs",
        )
    else:
        # No space, just fixed hyperparameters
        config.update(
            {
                "thresh2": 0.8,
                "alpha_v0": 1.0,
                "alpha_v1": 1.0,
                "alpha_v2": 1.0,
                "alpha_t0": 1.0,
                "alpha_t1": 1.0,
                "alpha_t2": 1.0,
                "alpha_thresh0": 1.0,
                "alpha_thresh1": 1.0,
                "tau_v0": 0.8,
                "tau_v1": 0.8,
                "tau_v2": 0.8,
                "tau_t0": 0.8,
                "tau_t1": 0.8,
                "tau_t2": 0.8,
                "tau_thresh0": 0.8,
                "tau_thresh1": 0.8,
                "a_pre": 1.0,
                "lr": 0.0001,
                "tau_e_trace": 0.8,
            }
        )

        # Run main
        main(config)
