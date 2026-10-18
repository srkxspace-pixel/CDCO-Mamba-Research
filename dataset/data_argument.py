# Reconstructed from the repo's dataset/__pycache__/data_argument.cpython-38.pyc
# (the .py was never committed). Behaviour kept identical, including that
# Dropout and random_erase_3d modify the original training tensors in place.
import numpy as np
import torch
import torch.nn as nn
from .preprocess import StandardScaler
from scipy.ndimage import rotate
import random


class Dropout(nn.Module):
    def __init__(self, p_zero=0.5):
        super(Dropout, self).__init__()
        self.p_zero = p_zero

    def forward(self, x_t):
        if not self.training or self.p_zero == 0:
            return x_t
        num_dim1, num_dim2 = x_t.size()
        num_drop = int(num_dim1 * self.p_zero)
        drop_indices = np.random.choice(num_dim1, num_drop, replace=False)
        x_t[drop_indices, :] = 0
        return x_t


def random_rotate_3d(tensor, angles_range=(-180, 180)):
    angle_x = random.uniform(*angles_range)
    angle_y = random.uniform(*angles_range)
    angle_z = random.uniform(*angles_range)
    tensor_np = tensor.squeeze(0).numpy()
    rotated_tensor_np = rotate(tensor_np, angle_x, axes=(1, 2), reshape=False, order=1)
    rotated_tensor_np = rotate(rotated_tensor_np, angle_y, axes=(0, 2), reshape=False, order=1)
    rotated_tensor_np = rotate(rotated_tensor_np, angle_z, axes=(0, 1), reshape=False, order=1)
    rotated_tensor = torch.from_numpy(rotated_tensor_np).unsqueeze(0)
    return rotated_tensor


def random_erase_3d(tensor, max_size_ratio=0.1):
    _, depth, height, width = tensor.shape
    erase_depth = random.randint(1, int(depth * max_size_ratio))
    erase_height = random.randint(1, int(height * max_size_ratio))
    erase_width = random.randint(1, int(width * max_size_ratio))
    z_start = random.randint(0, depth - erase_depth)
    y_start = random.randint(0, height - erase_height)
    x_start = random.randint(0, width - erase_width)
    tensor[:, z_start:z_start + erase_depth, y_start:y_start + erase_height, x_start:x_start + erase_width] = 0
    return tensor


def random_erase_3d_checkerboard(tensor, drop_ratio=0.25):
    batch, depth, height, width = tensor.shape
    patch_size = depth // 24
    for z in range(0, depth, patch_size):
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                if random.random() < drop_ratio:
                    tensor[:, z:z + patch_size, y:y + patch_size, x:x + patch_size] = 0
    return tensor


def data_aug(time, pc, label, smri, cache):
    """Upstream data_aug, except the 4x sMRI set goes to a .npy memmap at `cache`.

    Upstream held it in RAM (~10 GB peak for ~230 training volumes; this machine has 8 GB).
    Repaired on this branch: upstream's Dropout and random_erase_3d modified `time`/`smri` in
    place, cumulatively zeroing ~87% of each original's ROI rows and leaving three identical
    doubly-erased copies of every volume. Here each augmented copy gets its own dropout/erasure
    and the originals stay clean. Dropped: the dead NaN `continue` (NaNs are zeroed on the line
    before) and the final joint randperm (the train DataLoader shuffles anyway).
    """
    augment_time = []
    augment_lable = []
    augment_pc = []

    length, _, _ = time.shape
    out = np.lib.format.open_memmap(cache, mode='w+', dtype=np.float32,
                                    shape=(4 * length,) + tuple(smri.shape[1:]))
    aug_times = 3
    for j in range(aug_times):
        for i in range(length):
            data = time[i].clone()
            label1 = label[i]
            smri1 = smri[i]

            aug_time = Dropout()(data)
            pcc = np.corrcoef(aug_time)
            aug_time = np.array(aug_time)

            pcc[np.isnan(pcc)] = 0

            augment_time.append(aug_time)
            augment_lable.append(label1)
            augment_pc.append(pcc)

            if j == 0:
                out[length + i] = random_rotate_3d(smri1).numpy()
            else:
                out[(j + 1) * length + i] = random_erase_3d(smri1.clone()).numpy()
    out[:length] = smri.numpy()
    out.flush()

    augment_time = torch.tensor(np.array(augment_time))
    augment_lable = torch.stack(augment_lable)
    augment_pc = torch.tensor(np.array(augment_pc))

    fianl_time = torch.cat([time, augment_time], dim=0)
    fianl_label = torch.cat([label, augment_lable], dim=0)
    fianl_pc = torch.cat([pc, augment_pc], dim=0)
    fianl_smri = torch.from_numpy(out)

    return fianl_time.float(), fianl_pc.float(), fianl_label, fianl_smri
