"""Define utility function."""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import transforms
import cv2
# import gym
# import gym.wrappers


######################################################
def make_dirs(directory_name):   # directory_name
    """Create a new directory."""
    for i, p_p in enumerate(directory_name.split("/")):
        p_p = "/".join(directory_name.split("/")[:i]) + "/" + p_p
        if not os.path.isdir(p_p):
            os.mkdir(p_p)


######################################################
def save_result(sdir, name, data):
    """Save the result."""
    plt.clf()
    plt.xlabel("Number of Epoch")
    plt.ylabel("Mean-square error")
    for key, val in data.items():
        plt.plot(range(len(val)), val, label=key)
    plt.legend()
    plt.savefig(sdir + name + ".pdf")
    pd.DataFrame(data).to_csv(sdir + name + ".csv")


######################################################
def save_image(sdir, data, name, n_repeat):
    """Save the Image."""
    plt.clf()
    array = data[0].to("cpu")
    array = array.data.numpy().astype("f")
    # array = array.transpose(1, 2, 0)
    array = array * 255.0
    # img = Image.fromarray(np.uint8(array))
    # draw = ImageDraw.Draw(img)
    # img.save(sdir + name +".png")
    plt.imshow(array[0], cmap="gray")
    plt.savefig(sdir + name + str(n_repeat) + ".png")


######################################################
def wait_zoom(env, action, is_view):
    """Wait for zoom in gym."""
    while True:
        # print(env.t)
        if is_view:
            env.render()
        # observation, reward, done, info = env.step(action)
        observation, _, _, _ = env.step(action)
        if env.t >= 1.0:
            break
    print("Ready to start!")
    return observation


def get_env_info(env):
    """Get state from gym."""
    s_dim = env.observation_space.shape[0] if \
        len(env.observation_space.shape) == 1 else (1, 32, 32)
    a_dim = env.action_space.shape[0]
    trans = None if isinstance(s_dim, int) else transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((32, 32)),
         transforms.Grayscale(),
         transforms.ToTensor()]
    )
    return s_dim, a_dim, trans


def init_policy(env, policy, scale_factor=2.0,
                dataset=None, n_epoch=10, n_sample=256, n_count=100):
    """Initialize the policy."""
    if "laplace" in policy.dist:
        try:
            loc = torch.from_numpy(
                (0.5 * (env.action_space.high + env.action_space.low)).
                astype(np.float32)).to(policy.device)
            scale = torch.from_numpy(
                (0.5 * (env.action_space.high - env.action_space.low)).
                astype(np.float32)).to(policy.device)
        except:
            loc = torch.zeros(policy.a_dim, dtype=np.float32,
                              device=policy.device)
            scale = torch.ones(policy.a_dim, dtype=np.float32,
                               device=policy.device)
        scale /= scale_factor * np.sqrt(2.0)
        prior = torch.distributions.laplace.Laplace(loc, scale)
    else:
        try:
            loc = torch.from_numpy(
                (0.5 * (env.action_space.high + env.action_space.low)).
                astype(np.float32)).to(policy.device)
            scale = torch.from_numpy(
                (0.5 * (env.action_space.high - env.action_space.low)).
                astype(np.float32)).to(policy.device)
        except:
            loc = torch.zeros(policy.a_dim, dtype=np.float32,
                              device=policy.device)
            scale = torch.ones(policy.a_dim, dtype=np.float32,
                               device=policy.device)
        scale /= scale_factor
        prior = torch.distributions.normal.Normal(loc, scale)
    policy.prior = prior

    def _init_policy(s_o):
        # a_c = policy(s_o)
        loss = policy.criterion_init().mean()
        policy.optimizer.zero_grad()
        loss.backward()
        policy.optimizer.step()
        return loss.data.item()

    if prior is not None:
        print(f"towards {prior.mean}, {prior.variance}")
        policy.train()
        for epch in range(1, n_epoch+1):
            loss_sum = 0.0
            loss_num = 0
            if dataset is not None:
                for batch_idx, (s_o, a_c) in enumerate(dataset):
                    loss_sum += _init_policy(s_o)
                    loss_num += 1
            else:
                for _ in range(n_count):
                    s_o = np.array([env.observation_space.sample().tolist()
                                   for _ in range(n_sample)])
                    loss_sum += _init_policy(s_o)
                    loss_num += 1
            print(f"{epch}-th epoch to initialize policy "
                  f"was end: \n\tloss = {loss_sum / loss_num}")


######################################################
def collect_data(env, policy, env_name, transform,
                 n_time, is_view, sdir_traj=None,
                 sdir_reward=None, sdir_action=None, file_name=None):
    """Collect data in GYM."""
    trajectory = []
    result = []
    actions = []
    observation = env.reset()
    action = policy.reset()

    # wait until zoom
    if "CarRacing" in env_name:
        observation = wait_zoom(env, action, is_view)
    # main loop to update trajectory
    for t_time in range(1, n_time+1):
        print(f'starting the {t_time}-th update')
        if is_view:
            env.render()
        obs = observation if transform is None else \
            transform(observation.astype(np.uint8))
        # print(obs)
        action = policy(obs)
        if "Tensor" in str(type(action)):
            action = action.cpu().data.numpy().flatten()
        if sdir_traj is not None:
            trajectory.append((np.asarray(observation, dtype=np.float32),
                               np.asarray(action, dtype=np.float32)))
        observation, reward, done, _ = env.step(action)
        if sdir_reward is not None:
            result.append(reward)
        if sdir_action is not None:
            actions.append(action.tolist())
        if done:
            break

    # record trajectory and return at the end of trajectory
    if file_name is None:
        file_name = datetime.now().strftime("%Y%m%d%H%M%S")
    rtv = []
    print(f"Finish one episode, and record it to {file_name}")
    if sdir_traj is not None:
        pd.to_pickle(trajectory, sdir_traj + file_name + ".gz")
        rtv.append(sdir_traj + file_name + ".gz")
    if sdir_reward is not None:
        np.savetxt(sdir_reward + file_name + ".csv",
                   np.array([result]).T, delimiter=",")
        rtv.append(np.sum(result))
    if sdir_action is not None:
        actions = np.array(actions)
        plt.clf()
        for idx in range(len(action)):
            sns.distplot(actions[:, idx], label=str(idx))
        plt.legend()
        plt.tight_layout()
        plt.savefig(sdir_action + file_name + ".pdf")
    return rtv


# Self crop image
def __crop(img, pos, size):
    """Crop the image."""
    o_w, o_h = img.size
    x_1, y_1 = pos
    t_w, t_h = size
    if t_w < o_w and t_h < o_h:
        return img.crop((x_1, y_1, x_1 + t_w, y_1 + t_h))
    else:
        raise ValueError("please check the crop setting")


# draw shadow on the img
def draw_shadow(img, thickness=10, blur=3,
                angle=np.pi/2, offset_x=0, offset_y=0, gamma=1.5):
    """Draw random shadow on the img."""
    mask = np.zeros([img.shape[0], img.shape[1], 1])
    length_line = 200  # the length of the line
    point1 = (int(length_line * np.cos(angle) + offset_x + img.shape[1] / 2),
              int(length_line * np.sin(angle) + offset_y + img.shape[0] / 2))
    point2 = (int(-length_line * np.cos(angle) + offset_x + img.shape[1] / 2),
              int(-length_line * np.sin(angle) + offset_y + img.shape[0] / 2))
    cv2.line(mask, point1, point2, (1.0), thickness)
    # draw a line with value 1, all the other places 0
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    mask = mask.reshape([img.shape[0], img.shape[1], 1])

    img2 = np.power(np.copy(img), gamma)

    img_merged = mask * img2 + (1.0 - mask) * img
    img_merged = img_merged.astype(np.uint8)
    return img_merged


def _augment_single_image_with_shadow(img, shadow_max_gamma):
    """Augment the image."""
    thickness = np.random.randint(10, 100)
    kernel_sizes = [3, 5, 7]
    blur = kernel_sizes[np.random.randint(0, len(kernel_sizes))]
    angle = np.random.uniform(low=0, high=np.pi)
    offset_x = np.random.randint(-120, 120)
    offset_y = np.random.randint(-40, 40)
    gamma = np.random.uniform(low=0, high=shadow_max_gamma)
    if np.random.rand() > 0.5:
        gamma = 1/(1 + gamma)   # the pic will be darker
    else:
        gamma = 1 + gamma   # the pic will be lighter

    img_merged = draw_shadow(img, thickness, blur,
                             angle, offset_x, offset_y, gamma)
    return img_merged


# it is the same with tf.image.per_image_standardization
def _image_standardization(img):
    """Standardize the image."""
    # input has the shape (N, C, H, W)
    img = img.view(img.shape[0], -1)
    # flatten img_channel * img_height * img_width (N, F)
    mean = torch.unsqueeze(torch.mean(img, dim=1), dim=1)
    # (mean of each row), expand to 2 dimension
    std = torch.unsqueeze(torch.std(img, dim=1), dim=1) + 1e-6
    # small term in order to avoid dividing by zero
    img = (img - mean) / std  # avoid dividing zero
    return img  # (return back (N, F), needs reshape to (N,C,H,W) in CNN layer)


def epoch_policy(dataset, policy, n_epoch, mode, network):
    """Calculate Training loss."""
    origin_loss_sum = 0.0
    loss_num = 0
    if "train" in mode:
        policy.train()
        # what is policy.train(), policy is the object of Class Policy
    else:
        policy.eval()

    for _, (s_o, a_c) in enumerate(dataset):
        # _ is batch_idx
        s_o = s_o.to(policy.device, non_blocking=True)
        a_c = a_c.to(policy.device, non_blocking=True)
        output = policy(s_o)

        origin_loss = policy.criterion(output, a_c)  # original loss
        if policy.training:
            # nn.modules中的attribute, boolean value,
            # indicate if train mode or not. (drop out when in train mode)
            policy.optimizer.zero_grad()
            # policy is an object of Class Policy,
            # Policy inherit from Abstract,
            origin_loss.backward()
            policy.optimizer.step()
            # after one step optimization on params,
            # needs to apply_constraints on the params,
            # which means w, sensory_w, cm, gleak should be greater than 0
            if network == 'NCP':
                policy.ltc_cell.apply_weight_constraints()

        origin_loss_sum += origin_loss.data.item()
        loss_num += 1

    origin_loss_sum /= loss_num
    # loss_num has the information of batches, so how many batches
    print(f"{n_epoch}-th epoch {mode} "
          f"of policy was end: \n\tloss = {origin_loss_sum}")
    return origin_loss_sum


# calculate loss (sequentially over a dataset)
def evaluate_on_single_sequence(number, dataset, policy, n_epoch, mode):
    """Calculate validation loss."""
    if mode != 'train':
        policy.eval()
    loss_num = 0
    origin_loss_sum = 0
    for i in range(number):
        rnn_state = None
        # reset the hidden state when starting a sequence
        for pic, action in dataset.iterate_as_single_sequence(i):
            # take each pic and action
            # NCP forward procedure, : evaluate_on_single_sequence
            # send to cuda GPU
            pic = pic.to(policy.device, non_blocking=True)
            action = action.to(policy.device, non_blocking=True)

            pic = pic.unsqueeze(dim=0)
            # change from (16,channel,width,height) to
            # (1,16,channel,width,height)
            action = action.unsqueeze(dim=0)  # from (16,3) to (1,16,3)
            action_predicted, rnn_state = policy.evaluate_on_single_sequence(
                pic, rnn_state)
            origin_loss = policy.criterion(action_predicted, action)

            origin_loss_sum += origin_loss.data.item()
            loss_num += 1
    origin_loss_sum /= loss_num
    print(f"{n_epoch}-th epoch {mode} of "
          f"policy was end: \n\tloss = {origin_loss_sum}")
    return origin_loss_sum
