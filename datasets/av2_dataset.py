from typing import List, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Av2Dataset(Dataset):
    def __init__(
            self,
            data_root: Path,
            split: str = None,
            num_historical_steps: int = 50,
            split_points: List[int] = [50],
            radius: float = 150.0,
            max_neighbors: int = None,
            use_knn: bool = False,
            logger=None,
    ):
        assert split_points[-1] == 50 and num_historical_steps <= 50
        assert split in ['train', 'val', 'test']
        super(Av2Dataset, self).__init__()
        self.data_folder = Path(data_root) / split
        self.file_list = sorted(list(self.data_folder.glob('*.pt')))
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = 0 if split == 'test' else 60
        self.split_points = split_points
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.use_knn = use_knn
        self.delta_t = 0.1

        if logger is not None:
            logger.info(f'data root: {data_root}/{split}, total number of files: {len(self.file_list)}')

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        data = torch.load(self.file_list[index])
        data = self.process(data)
        return data

    def process(self, data):
        sequence_data = []
        for cur_step in self.split_points:
            ag_dict = self.process_single_agent(data, cur_step)
            sequence_data.append(ag_dict)
        return sequence_data

    def process_single_agent(self, data, step=50):
        idx = data['focal_idx']
        cur_agent_id = data['agent_ids'][idx]
        origin = data['x_positions'][idx, step - 1]
        theta = data['x_angles'][idx, step - 1]
        rotate_mat = self._build_rotation(-theta)

        ag_mask = self._select_neighbors(data, origin, idx, step)

        # transform agents to local
        st, ed = step - self.num_historical_steps, step + self.num_future_steps
        attr = torch.cat([data['x_attr'][[idx]], data['x_attr'][ag_mask]])
        pos = data['x_positions'][:, st:ed]
        pos = torch.cat([pos[[idx]], pos[ag_mask]])
        head = data['x_angles'][:, st:ed]
        head = torch.cat([head[[idx]], head[ag_mask]])
        valid_mask = data['x_valid_mask'][:, st:ed].bool()
        valid_mask = torch.cat([valid_mask[[idx]], valid_mask[ag_mask]])

        pos_hist = pos[:, :self.num_historical_steps]
        head_hist = head[:, :self.num_historical_steps]
        valid_hist = valid_mask[:, :self.num_historical_steps]

        pos_hist = self._transform_positions(pos_hist, origin, rotate_mat, valid_hist)
        vel_hist = self._compute_velocity(pos_hist, valid_hist)
        acc_hist = self._compute_acceleration(vel_hist, valid_hist)
        head_hist = self._normalize_headings(head_hist, theta, valid_hist)
        head_feat = torch.stack([torch.sin(head_hist), torch.cos(head_hist)], dim=-1)
        head_feat = torch.where(valid_hist.unsqueeze(-1), head_feat, torch.zeros_like(head_feat))

        size, obj_type = self._split_attr(attr)
        time_encoding = self._build_time_encoding(valid_hist.size(1), pos_hist.device, pos_hist.dtype)
        time_encoding = time_encoding.unsqueeze(0).expand(pos_hist.size(0), -1, -1)
        time_encoding = torch.where(valid_hist.unsqueeze(-1), time_encoding, torch.zeros_like(time_encoding))

        pos_hist = torch.where(valid_hist.unsqueeze(-1), pos_hist, torch.zeros_like(pos_hist))
        vel_hist = torch.where(valid_hist.unsqueeze(-1), vel_hist, torch.zeros_like(vel_hist))
        acc_hist = torch.where(valid_hist.unsqueeze(-1), acc_hist, torch.zeros_like(acc_hist))

        pos_ctr = pos_hist[:, -1].clone()

        if self.num_future_steps > 0:
            pos_future = pos[:, self.num_historical_steps:]
            valid_future = valid_mask[:, self.num_historical_steps:]
            pos_future = self._transform_positions(pos_future, origin, rotate_mat, valid_future)
            target = torch.where(
                valid_future.unsqueeze(-1), pos_future - pos_ctr.unsqueeze(1), torch.zeros_like(pos_future)
            )
            target_mask = valid_future
        else:
            target = target_mask = None

        lane_features = self._process_lanes(data, origin, rotate_mat)

        return {
            'target': target,
            'target_mask': target_mask,
            'agents': {
                'positions': pos_hist,
                'velocities': vel_hist,
                'accelerations': acc_hist,
                'orientations': head_feat,
                'type': obj_type,
                'size': size,
                'valid_mask': valid_hist,
                'time_encoding': time_encoding,
                'focal_agent_idx': torch.tensor(0, dtype=torch.long),
            },
            'lane_features': lane_features,
            'origin': origin.view(1, 2),
            'theta': theta.view(1),
            'scenario_id': data['scenario_id'],
            'track_id': cur_agent_id,
            'city': data['city'],
            'timestamp': torch.Tensor([step * self.delta_t])
        }

    def _build_rotation(self, theta: torch.Tensor) -> torch.Tensor:
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        first_row = torch.stack([cos_theta, -sin_theta])
        second_row = torch.stack([sin_theta, cos_theta])
        rotate_mat = torch.stack([first_row, second_row])
        return rotate_mat

    def _select_neighbors(self, data, origin, idx, step) -> torch.Tensor:
        final_positions = data['x_positions'][:, step - 1]
        valid_agents = data['x_valid_mask'][:, step - 1].bool()
        distances = torch.norm(final_positions - origin, dim=-1)

        neighbor_mask = torch.zeros_like(distances, dtype=torch.bool)
        if self.use_knn and self.max_neighbors is not None and self.max_neighbors > 0:
            indices = torch.arange(distances.numel(), device=distances.device)
            valid_idx = torch.nonzero(valid_agents & (indices != idx))[:, 0]
            if valid_idx.numel() > 0:
                valid_distances = distances[valid_idx]
                k = min(self.max_neighbors, valid_idx.numel())
                topk = torch.topk(valid_distances, k=k, largest=False).indices
                selected = valid_idx[topk]
                neighbor_mask[selected] = True
        else:
            neighbor_mask = distances < self.radius
            neighbor_mask &= valid_agents
            neighbor_mask[idx] = False
            if self.max_neighbors is not None and self.max_neighbors > 0:
                candidate_idx = torch.nonzero(neighbor_mask)[:, 0]
                if candidate_idx.numel() > self.max_neighbors:
                    candidate_dist = distances[candidate_idx]
                    order = torch.argsort(candidate_dist)
                    keep = candidate_idx[order[: self.max_neighbors]]
                    neighbor_mask = torch.zeros_like(neighbor_mask, dtype=torch.bool)
                    neighbor_mask[keep] = True

        neighbor_mask[idx] = False
        return neighbor_mask

    def _transform_positions(self, pos, origin, rotate_mat, valid_mask):
        num_agents, num_steps = pos.shape[:2]
        pos_local = pos - origin.view(1, 1, -1)
        reshaped = pos_local.view(-1, 2).t()
        rotated = torch.matmul(rotate_mat, reshaped).t().view(num_agents, num_steps, 2)
        rotated = torch.where(valid_mask.unsqueeze(-1), rotated, torch.zeros_like(rotated))
        return rotated

    def _normalize_headings(self, head, theta, valid_mask):
        head = (head - theta + np.pi) % (2 * np.pi) - np.pi
        head = torch.where(valid_mask, head, torch.zeros_like(head))
        return head

    def _compute_velocity(self, pos, valid_mask):
        vel = torch.zeros_like(pos)
        if pos.size(1) < 2:
            return vel
        diff_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        pos_diff = pos[:, 1:] - pos[:, :-1]
        vel[:, 1:] = torch.where(diff_mask.unsqueeze(-1), pos_diff / self.delta_t, torch.zeros_like(pos_diff))
        return vel

    def _compute_acceleration(self, vel, valid_mask):
        acc = torch.zeros_like(vel)
        if vel.size(1) < 2:
            return acc
        vel_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
        vel_diff = vel[:, 1:] - vel[:, :-1]
        acc[:, 1:] = torch.where(vel_valid.unsqueeze(-1), vel_diff / self.delta_t, torch.zeros_like(vel_diff))
        return acc

    def _split_attr(self, attr) -> Tuple[torch.Tensor, torch.Tensor]:
        if attr.size(-1) == 0:
            size = torch.zeros(attr.shape[:-1] + (0,), dtype=attr.dtype)
            obj_type = torch.full(attr.shape[:-1], -1, dtype=torch.long)
        else:
            size = attr[..., :-1]
            obj_type = attr[..., -1].long()
        return size, obj_type

    def _build_time_encoding(self, num_steps: int, device, dtype):
        time_steps = torch.arange(-(num_steps - 1), 1, device=device, dtype=dtype) * self.delta_t
        period = max(num_steps * self.delta_t, self.delta_t)
        angles = 2 * np.pi * time_steps / period
        return torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)

    def _process_lanes(self, data, origin, rotate_mat):
        lane_pos = data['lane_positions']
        lane_attr = data['lane_attr']
        lane_is_int = data['is_intersections']

        lane_pos = lane_pos.reshape(-1, lane_pos.size(1), 2)
        lane_pos = lane_pos - origin.view(1, 1, -1)
        reshaped = lane_pos.reshape(-1, 2).t()
        lane_pos = torch.matmul(rotate_mat, reshaped).t().reshape(-1, lane_pos.size(1), 2)

        valid_mask = (
            (lane_pos[:, :, 0] > -self.radius)
            & (lane_pos[:, :, 0] < self.radius)
            & (lane_pos[:, :, 1] > -self.radius)
            & (lane_pos[:, :, 1] < self.radius)
        )

        lane_mask = valid_mask.any(dim=-1)
        lane_pos = lane_pos[lane_mask]
        lane_attr = lane_attr[lane_mask]
        lane_is_int = lane_is_int[lane_mask]
        valid_mask = valid_mask[lane_mask]

        resampled_pos, resampled_mask = self._resample_centerlines(lane_pos, valid_mask, num_points=24)
        directions = self._compute_directions(resampled_pos, resampled_mask)
        curvature = self._compute_curvature(resampled_pos, resampled_mask)
        arc_length, arc_encoding = self._compute_arc_encoding(resampled_pos, resampled_mask)

        return {
            'positions': resampled_pos,
            'directions': directions,
            'curvature': curvature,
            'semantic': lane_attr,
            'is_intersection': lane_is_int,
            'valid_mask': resampled_mask,
            'arc_length': arc_length,
            'arc_encoding': arc_encoding,
        }

    def _resample_centerlines(self, lane_pos, valid_mask, num_points=24):
        device = lane_pos.device
        dtype = lane_pos.dtype
        resampled = []
        resampled_mask = []
        for pos, mask in zip(lane_pos, valid_mask):
            valid_points = pos[mask]
            if valid_points.size(0) < 2:
                new_pos = torch.zeros(num_points, 2, device=device, dtype=dtype)
                new_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
            else:
                distances = torch.norm(valid_points[1:] - valid_points[:-1], dim=-1)
                cumulative = torch.cat([torch.zeros(1, device=device, dtype=dtype), torch.cumsum(distances, dim=0)])
                total_length = cumulative[-1]
                if total_length < 1e-6:
                    new_pos = valid_points[0].repeat(num_points, 1)
                    new_mask = torch.ones(num_points, dtype=torch.bool, device=device)
                else:
                    sample_steps = torch.linspace(0, total_length, num_points, device=device, dtype=dtype)
                    new_pos = self._interpolate_polyline(valid_points, cumulative, sample_steps)
                    new_mask = torch.ones(num_points, dtype=torch.bool, device=device)
            resampled.append(new_pos)
            resampled_mask.append(new_mask)

        return torch.stack(resampled, dim=0) if resampled else torch.zeros(0, num_points, 2, device=device, dtype=dtype), \
            torch.stack(resampled_mask, dim=0) if resampled_mask else torch.zeros(0, num_points, dtype=torch.bool, device=device)

    def _interpolate_polyline(self, points, cumulative, samples):
        idxs = torch.searchsorted(cumulative, samples, right=True).clamp(max=cumulative.numel() - 1)
        idxs0 = torch.clamp(idxs - 1, min=0)
        idxs1 = idxs
        denom = (cumulative[idxs1] - cumulative[idxs0]).clamp(min=1e-6)
        t = (samples - cumulative[idxs0]) / denom
        interpolated = points[idxs0] + (points[idxs1] - points[idxs0]) * t.unsqueeze(-1)
        return interpolated

    def _compute_directions(self, positions, valid_mask):
        if positions.size(1) < 2:
            return torch.zeros_like(positions)
        diff = torch.zeros_like(positions)
        diff[:, 1:] = positions[:, 1:] - positions[:, :-1]
        diff[:, 0] = diff[:, 1]
        diff = torch.where(valid_mask.unsqueeze(-1), diff, torch.zeros_like(diff))
        return diff

    def _compute_curvature(self, positions, valid_mask):
        num_points = positions.size(1)
        curvature = torch.zeros(positions.size(0), num_points, device=positions.device, dtype=positions.dtype)
        if num_points < 3:
            return curvature
        diff = positions[:, 1:] - positions[:, :-1]
        diff2 = diff[:, 1:] - diff[:, :-1]
        tangent = torch.zeros_like(positions)
        tangent[:, 1:] = diff
        tangent = torch.where(valid_mask.unsqueeze(-1), tangent, torch.zeros_like(tangent))
        numerator = tangent[:, :-1, 0] * diff2[:, :, 1] - tangent[:, :-1, 1] * diff2[:, :, 0]
        denominator = (tangent[:, :-1, 0] ** 2 + tangent[:, :-1, 1] ** 2).clamp(min=1e-6) ** 1.5
        curvature[:, 1:-1] = torch.zeros_like(curvature[:, 1:-1])
        curvature[:, 1:-1] = numerator / denominator
        curvature = torch.where(valid_mask, curvature, torch.zeros_like(curvature))
        return curvature

    def _compute_arc_encoding(self, positions, valid_mask):
        if positions.size(1) == 0:
            return (
                torch.zeros_like(positions[..., 0]),
                torch.zeros(positions.size(0), 0, 2, device=positions.device, dtype=positions.dtype),
            )
        distances = torch.norm(positions[:, 1:] - positions[:, :-1], dim=-1)
        cumulative = torch.zeros(positions.size(0), positions.size(1), device=positions.device, dtype=positions.dtype)
        cumulative[:, 1:] = torch.cumsum(distances, dim=-1)
        total = cumulative.gather(-1, valid_mask.long().sum(-1, keepdim=True).clamp(min=1) - 1)
        total = total.clamp(min=self.delta_t)
        normalized = torch.where(
            valid_mask,
            cumulative / total,
            torch.zeros_like(cumulative),
        )
        encoding = torch.stack([
            torch.sin(2 * np.pi * normalized),
            torch.cos(2 * np.pi * normalized),
        ], dim=-1)
        encoding = torch.where(valid_mask.unsqueeze(-1), encoding, torch.zeros_like(encoding))
        return cumulative, encoding


def collate_fn(seq_batch):
    seq_data = []
    for i in range(len(seq_batch[0])):
        batch = [b[i] for b in seq_batch]
        data = {}

        agent_batch = [b['agents'] for b in batch]
        agents = {}
        for key in ['positions', 'velocities', 'accelerations', 'orientations', 'time_encoding', 'size']:
            agents[key] = pad_sequence([a[key] for a in agent_batch], batch_first=True)

        agents['type'] = pad_sequence(
            [a['type'] for a in agent_batch], batch_first=True, padding_value=-1
        )
        agents['valid_mask'] = pad_sequence(
            [a['valid_mask'] for a in agent_batch], batch_first=True, padding_value=False
        )
        agents['focal_agent_idx'] = torch.stack(
            [a['focal_agent_idx'].clone() for a in agent_batch]
        )

        data['agents'] = agents

        lane_batch = [b['lane_features'] for b in batch]
        lane = {}
        for key in ['positions', 'directions', 'arc_encoding']:
            lane[key] = pad_sequence([l[key] for l in lane_batch], batch_first=True)

        lane['curvature'] = pad_sequence(
            [l['curvature'] for l in lane_batch], batch_first=True
        )
        lane['arc_length'] = pad_sequence(
            [l['arc_length'] for l in lane_batch], batch_first=True
        )
        lane['semantic'] = pad_sequence(
            [l['semantic'] for l in lane_batch], batch_first=True
        )
        lane['is_intersection'] = pad_sequence(
            [l['is_intersection'] for l in lane_batch], batch_first=True, padding_value=False
        )
        lane['valid_mask'] = pad_sequence(
            [l['valid_mask'] for l in lane_batch], batch_first=True, padding_value=False
        )

        data['lane_features'] = lane

        if batch[0]['target'] is not None:
            data['target'] = pad_sequence([b['target'] for b in batch], batch_first=True)
            data['target_mask'] = pad_sequence(
                [b['target_mask'] for b in batch], batch_first=True, padding_value=False
            )

        data['scenario_id'] = [b['scenario_id'] for b in batch]
        data['track_id'] = [b['track_id'] for b in batch]
        data['city'] = [b['city'] for b in batch]

        data['origin'] = torch.cat([b['origin'] for b in batch], dim=0)
        data['theta'] = torch.cat([b['theta'] for b in batch])
        data['timestamp'] = torch.cat([b['timestamp'] for b in batch])
        seq_data.append(data)
    return seq_data
