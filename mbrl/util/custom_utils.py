import numpy as np

# def report_grad_norm(self):
#         # may add qf1, policy, etc.
#         policy = self.locals['self'].policy
#         grads = {}
#         grads["actor"] = self.get_grad_norm(policy.actor)
#         grads["critic"] = self.get_grad_norm(policy.critic)
#         grads["critic_target"] = self.get_grad_norm(policy.critic_target)
#         if policy.features_extractor_class is not None:
#             grads["actor_encoder"] = self.get_grad_norm(policy.actor.features_extractor)
#             grads["critic_encoder"] = self.get_grad_norm(policy.critic.features_extractor)
#         return grads

def get_grad_norm(model):
    grad_norm = []
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        grad_norm.append(p.grad.data.norm(2).item())
    if grad_norm:
        grad_norm = np.mean(grad_norm)
    else:
        grad_norm = 0.0
    return grad_norm

ROLLOUT_LOG_FORMAT = [
    ("obs_mean", "OMe", "float"),
    ("obs_std", "OS", "float"),
    ("obs_min", "OMi", "float"),
    ("obs_max", "OMa", "float"),
    ("rew_mean", "RMe", "float"),
    ("success_rate", "SR", "float"),
]

class CustomLogHandler():
    def __init__(self, logger):
        self.logger = logger
        self.imag_env_steps = 0

        self.init_rollout_tracking()

    def init_rollout_tracking(self):
        self.logger.register_group(
            "rollout_real",
            ROLLOUT_LOG_FORMAT,
            color="green",
            dump_frequency=1,
        )
        self.logger.register_group(
            "rollout_imag",
            ROLLOUT_LOG_FORMAT,
            color="blue",
            dump_frequency=1,
        )

    def log_rollout(self, env_steps, obs_list, rew_list, action_list, success_list=None, type='rollout_real', imag_steps=None):
        assert type in ['rollout_real', 'rollout_imag']
        
        obs_mean, obs_max, obs_min, obs_std = np.mean(obs_list), np.max(obs_list), np.min(obs_list), np.std(obs_list)

        log_data = {}
        log_data["obs_mean"] = obs_mean
        log_data["obs_max"] = obs_max
        log_data["obs_min"] = obs_min
        log_data["obs_std"] = obs_std
        # self.logger.log(f"{type}/obs_mean", obs_mean, env_steps)
        # self.logger.log(f"{type}/obs_max", obs_max, env_steps)
        # self.logger.log(f"{type}/obs_min", obs_min, env_steps)
        # self.logger.log(f"{type}/obs_std", obs_std, env_steps)

        act_mean, act_max, act_min, act_std = np.mean(action_list), np.max(action_list), np.min(action_list), np.std(action_list)
        log_data["act_mean"] = act_mean
        log_data["act_max"] = act_max
        log_data["act_min"] = act_min
        log_data["act_std"] = act_std

        rew_mean = np.mean(rew_list)
        # self.logger.log(f"{type}/rew_mean", rew_mean, env_steps)
        log_data["rew_mean"] = rew_mean

        if success_list is not None:
            success_rate = np.mean(success_list)
            # self.logger.log(f"{type}/success_rate", success_rate, env_steps)
            log_data["success_rate"] = success_rate

        # self.logger.log(f"{type}/env_steps", env_steps, env_steps)
        log_data["env_steps"] = env_steps

        if type == 'rollout_imag':
            self.imag_env_steps += imag_steps
            # self.logger.log(f"{type}/total_imag_env_steps", self.imag_env_steps, env_steps)
            # self.logger.log(f"{type}/iter_imag_steps", imag_steps, env_steps)
            log_data["total_imag_env_steps"] = self.imag_env_steps
            log_data["iter_imag_steps"] = imag_steps

        self.logger.log_data(type, log_data)
        

#######################################
# Video

from mbrl.third_party.pytorch_sac import utils as vid_utils
import imageio
import os

class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, camera_id=0, fps=30):
        self.save_dir = vid_utils.make_dir(root_dir, "video") if root_dir else None
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render()
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)

#####################################