"""Load parameters.

This file is used to check the saved setting.pth,
where all the information of training parameters,
and network details are saved.
"""

import torch

PATH = "/Users/haotianxue/Desktop/settings.pth"

load_state_dict = torch.load(PATH, map_location=torch.device('cpu'))
print(load_state_dict)
