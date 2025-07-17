import json
import os
import shutil
import numpy as np
import torch
import random
import torch.nn as nn


def init_weight(weight, args):
    if args.init == "uniform":
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == "normal":
        nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m, args):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            init_weight(m.weight, args)
        if hasattr(m, "bias") and m.bias is not None:
            init_bias(m.bias)
    elif classname.find("Downsampler") != -1:
        if hasattr(m, "leftmost_group"):
            init_weight(m.leftmost_group, args)
        if hasattr(m, "r_w_bias"):
            init_weight(m.r_w_bias, args)
        if hasattr(m, "r_r_bias"):
            init_weight(m.r_r_bias, args)
    elif classname.find("Embedding") != -1:
        if hasattr(m, "weight"):
            init_weight(m.weight, args)
    elif classname.find("LayerNorm") != -1:
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, "bias") and m.bias is not None:
            init_bias(m.bias)
    elif classname.find("TransformerLM") != -1:
        if hasattr(m, "r_w_bias"):
            init_weight(m.r_w_bias, args)
        if hasattr(m, "r_r_bias"):
            init_weight(m.r_r_bias, args)


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_mean_with_padding(batch_tensor, batch_masks):
    expanded_masks_batch = batch_masks.unsqueeze(-1).expand_as(batch_tensor)
    masked_tensor = batch_tensor * expanded_masks_batch
    sum_tensor = masked_tensor.sum(dim=1)
    count_tensor = (expanded_masks_batch != 0).sum(dim=1)
    mean_tensor = sum_tensor / count_tensor

    return mean_tensor


def read_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        return

    os.makedirs(dir_path, exist_ok=True)

    print("Experiment dir : {}".format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, "scripts")
        os.makedirs(script_path, exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_args_to_json(args, folder_path):
    args_dict = vars(args)
    with open(os.path.join(folder_path, "config.json"), "w") as json_file:
        json.dump(args_dict, json_file, indent=4)

    print("Arguments saved")


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(path):
    if os.path.isdir(path):
        path = os.path.join(path, "checkpoint_last.pt")

    dst = f"cuda:{torch.cuda.current_device()}"
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=dst)
    return checkpoint


def save_ckpt(model, optimizer, path, epoch):
    torch.save(model, os.path.join(path, "model_{}.pt".format(epoch)))
    torch.save(
        optimizer.state_dict(), os.path.join(path, "optimizer_{}.pt".format(epoch))
    )


def calculate_mean(data_dict):
    """
    Calculate the mean for each key in a defaultdict.
    """
    mean_dict = {}
    for key, values in data_dict.items():
        if isinstance(
            values[0], torch.Tensor
        ):  # Check if the first value is a PyTorch tensor
            mean_dict[key] = torch.stack(values).mean(dim=0).item()
        else:
            mean_dict[key] = sum(values) / len(values)
    return mean_dict


def get_model_config(config, model_class):
    import inspect

    model_args = inspect.getfullargspec(model_class).args
    assert model_args.index("self") == 0
    model_args = model_args[1:]
    try:
        values = {arg: getattr(config, arg) for arg in model_args}
    except:
        values = {arg: config[arg] for arg in model_args if arg in config}
    return values


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm
