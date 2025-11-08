import torch
import random
import numpy as np
import torch.nn.functional as F



selectable_directions = [[[5, 6], [1, 2], [2, 3], [4, 5], [7, 8], [8, 9]],
                         [[5, 2], [4, 1], [6, 3], [7, 4], [8, 5], [9, 6]],
                         [[5, 3], [4, 2], [7, 5], [8, 6]],
                         [[5, 1], [6, 2], [8, 4], [9, 5]],
                         [[9, 4], [6, 1]],
                         [[4, 3], [7, 6]],
                         [[3, 8], [2, 7]],
                         [[8, 1], [9, 2]]]


class I2E(torch.nn.Module):
    def __init__(self, padding_mode='replicate', ratio=0.12, shuffle=[4, 5, 6, 7, 0, 1, 2, 3]):
        super().__init__()

        self.padding_mode = padding_mode
        self.ratio = ratio
        self.shuffle = np.array(shuffle)
        self.weight = torch.nn.Parameter(torch.zeros((8, 1, 3, 3)))
        self.weight.requires_grad_(False)
        self.set_weight()
        
    def set_weight(self):
        torch.nn.init.constant_(self.weight, 0.)
        selected_directions = [selectable_directions[i][0] for i in range(8)]
        for i, directions in enumerate(selected_directions):
            x0, y0 = (directions[0]-1) // 3, (directions[0]-1) % 3
            x1, y1 = (directions[1]-1) // 3, (directions[1]-1) % 3
            self.weight.data[i, 0, x0, y0] = -1
            self.weight.data[i, 0, x1, y1] = 1
        self.weight.data = self.weight.data[self.shuffle]

    def forward(self, img):                                                 # [B, 3, H, W]
        # self.set_weight()
        img_v, _ = torch.max(img, 1, keepdim=True)                          # [B, 1, H, W]
        img_v = F.pad(img_v, (1, 1, 1, 1), mode=self.padding_mode)          # [B, 1, H+2, W+2]
        img_range = (torch.max(torch.max(img_v, 3, True)[0], 2, True)[0] - torch.min(torch.min(img_v, 3, True)[0], 2, True)[0]).unsqueeze(0)  # [1, B, 1, 1, 1]
        all_img_v = F.conv2d(img_v, self.weight).permute(1, 0, 2, 3)        # [8, B, H, W]
        
        pos_v = all_img_v * all_img_v.ge(0).float()
        neg_v = -all_img_v * all_img_v.lt(0).float()
        out = torch.stack([pos_v, neg_v], dim=2)                            # [8, B, 2, H, W]
        out = out.ge(img_range*self.ratio).float()
        return out.detach()