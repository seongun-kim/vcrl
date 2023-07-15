from mimetypes import init
import os

from collections import OrderedDict
from numbers import Number

import numpy as np
from gym.spaces import Box

import scipy.signal
from multiworld.envs.pygame.walls import VerticalWall, HorizontalWall

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{} {}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats


def get_generic_path_information(paths, stat_prefix=''):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]

    rewards = np.vstack([path["rewards"] for path in paths])
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))
    actions = [path["actions"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"] for path in paths])
    else:
        actions = np.vstack([path["actions"] for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))
    statistics['Num Paths'] = len(paths)

    return statistics


def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)


def get_path_lengths(paths):
    return [len(path['observations']) for path in paths]


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]


def get_asset_full_path(file_name):
    return os.path.join(ENV_ASSET_DIR, file_name)


def concatenate_box_spaces(*spaces):
    """
    Assumes dtypes of all spaces are the of the same type
    """
    low = np.concatenate([space.low for space in spaces])
    high = np.concatenate([space.high for space in spaces])
    return Box(low=low, high=high, dtype=np.float32)


# def init_grid(grid):
#     # Grid to numpy array.
#     cell_size = 2. / 13

#     grid_split = grid.split('\n')
#     grid_height, grid_width = len(grid_split) - 2, len(grid_split[1])

#     grid = grid.replace('\n', '')
#     grid_S = (np.array(list(grid)) != 'S').reshape((grid_height, grid_width))
#     start_ind, = np.argwhere(grid_S == False)
#     grid_G = (np.array(list(grid)) != 'G').reshape((grid_height, grid_width))
#     # goal_ind, = np.argwhere(grid_G == False)
#     goal_inds = np.argwhere(grid_G == False)

#     grid = grid.replace('S', ' ')
#     grid = grid.replace('G', ' ')
#     grid = (np.array(list(grid)) != ' ').reshape((grid_height, grid_width)).astype(np.int64)
#     grid_wall_index = np.argwhere(grid == True)
#     grid_free_index = np.argwhere(grid != True)

#     extent=[
#         - 0.5 * grid_width * cell_size,
#         0.5 * grid_width * cell_size,
#         -0.5 * grid_height * cell_size,
#         0.5 * grid_height * cell_size]

#     return grid, extent, start_ind, goal_inds

def init_grid(grid, wall_thickness):
    # Grid to numpy array.
    # cell_size = 2. / 13
    cell_size = wall_thickness * 2

    grid_split = grid.split('\n')
    grid_height, grid_width = len(grid_split) - 2, len(grid_split[1])

    grid = grid.replace('\n', '')
    grid_S = (np.array(list(grid)) != 'S').reshape((grid_height, grid_width))
    start_ind, = np.argwhere(grid_S == False)
    grid_G = (np.array(list(grid)) != 'G').reshape((grid_height, grid_width))
    # goal_ind, = np.argwhere(grid_G == False)
    goal_inds, = np.argwhere(grid_G == False)

    grid = grid.replace('S', ' ')
    grid = grid.replace('G', ' ')
    grid = (np.array(list(grid)) != ' ').reshape((grid_height, grid_width)).astype(np.int64)
    grid_wall_index = np.argwhere(grid == True)
    grid_free_index = np.argwhere(grid != True)

    extent=[
        - 0.5 * grid_width * cell_size,
        0.5 * grid_width * cell_size,
        -0.5 * grid_height * cell_size,
        0.5 * grid_height * cell_size]

    grid_info = {
        'grid': grid,
        'extent': extent,
        'grid_width': grid_width,
        'grid_height': grid_height,
        'cell_size': cell_size
    }
    return grid_info

def filter_wall(grid_map, is_horizontal=False):
    # Filter wall vertical/horizontal.
    filter = np.zeros((3, 3))
    if is_horizontal:
        filter[1, :] = 1.
    else:
        filter[:, 1] = 1.
    filtered_map = scipy.signal.convolve2d(grid_map, filter, mode='same')
    filtered_map = np.where(filtered_map >= 2, 1, 0)
    return filtered_map


def cluster_wall(map, is_horizontal=False):
    if not is_horizontal:
        map = map.T

    # Index of walls.
    idx_i, idx_j = np.where(map > 0)
    idx = np.c_[idx_i, idx_j]

    # 1. Group by i-th index.
    clusters = np.split(idx, np.unique(idx[:, 0], return_index=True)[1][1:])

    # 2. Group by j-th index: Check whether j-th index is consecutive.
    walls = []
    for cluster in clusters:
        if is_horizontal:
            walls.extend(_consecutive(cluster, idx=1, stepsize=1))  
        else:
            tmp = _consecutive(cluster, idx=1, stepsize=1)
            tmp = np.c_[tmp[0][:, 1], tmp[0][:, 0]]
            # print(tmp)
            # exit()
            walls.append(tmp)
    walls = np.array(walls)

    return walls


def index_to_position(walls, idx_min, idx_max, boundary_min, boundary_max, ball_radius, wall_thickness, is_horizontal):
    '''
    walls: list of indices of each wall.
    HorizontalWall(min_dist, y_pos, left_x, right_x, thickness=0.)
    VerticalWall(min_dist, x_pos, bottom_y, top_y, thickness=0.)
    '''
    ret = []
    # TODO: Normalize position with argument.
    for idx, wall in enumerate(walls):
        if is_horizontal:
            y_pos = normalize_position(wall[0][0], idx_min, idx_max, boundary_min, boundary_max)
            left_x = normalize_position(wall[0][1], idx_min, idx_max, boundary_min, boundary_max)
            right_x = normalize_position(wall[-1][1], idx_min, idx_max, boundary_min, boundary_max)
            ret.append(HorizontalWall(ball_radius, y_pos, left_x, right_x, wall_thickness))
        else:
            x_pos = normalize_position(wall[0][1], idx_min, idx_max, boundary_min, boundary_max)
            bottom_y = normalize_position(wall[0][0], idx_min, idx_max, boundary_min, boundary_max)
            top_y = normalize_position(wall[-1][0], idx_min, idx_max, boundary_min, boundary_max)
            ret.append(VerticalWall(ball_radius, x_pos, bottom_y, top_y, wall_thickness))

    return ret


def normalize_position(pos, idx_min, idx_max, boundary_min, boundary_max):
    # Currently, assume that the wall is square.
    orig_range = idx_max - idx_min
    tgt_range = boundary_max - boundary_min

    return (pos - idx_min) / orig_range * tgt_range + boundary_min

def _consecutive(data, idx=1, stepsize=1):
    # Ref: https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
    # idx=0: horizontal
    # idx=1: vertical
    return np.split(data, np.where(np.diff(data[:, idx]) != stepsize)[0]+1)


# def grid_to_wall(grid, boundary_dist, ball_radius, wall_thickness):
#     grid_map, _, _, _ = init_grid(grid)

#     horizontal_map = filter_wall(grid_map, is_horizontal=True)
#     vertical_map = filter_wall(grid_map, is_horizontal=False)

#     horizontal_walls = cluster_wall(horizontal_map, is_horizontal=True)
#     vertical_walls = cluster_wall(vertical_map, is_horizontal=False)

#     boundary_min = -boundary_dist - ball_radius
#     boundary_max = boundary_dist + ball_radius

#     walls = []
#     walls.extend(index_to_position(horizontal_walls, 0, grid_map.shape[0]-1, boundary_min, boundary_max, ball_radius, wall_thickness, is_horizontal=True))
#     walls.extend(index_to_position(vertical_walls, 0, grid_map.shape[0]-1, boundary_min, boundary_max, ball_radius, wall_thickness, is_horizontal=False))

#     return walls


def grid_to_wall(grid, boundary_dist, ball_radius, wall_thickness):
    def hw_to_xy(h, w, grid_info):
        x = (w + 0.5) / grid_info['grid_width'] * (grid_info['extent'][1] - grid_info['extent'][0]) + grid_info['extent'][0]
        y = (grid_info['grid_height'] - h - 0.5) / grid_info['grid_height'] * (grid_info['extent'][3] - grid_info['extent'][2]) + grid_info['extent'][2]
        return x, y

    grid_info = init_grid(grid, wall_thickness)
    walls = []
    for h in range(grid_info['grid'].shape[0]):
        for w in range(grid_info['grid'].shape[1]):
            if grid_info['grid'][h, w] == 1:
                x, y = hw_to_xy(h, w, grid_info)
                walls.append(VerticalWall(ball_radius, x , y , y , wall_thickness))
    return walls


def plot_walls(walls, ax=None, axis_off=False):
    min_x = min([wall.min_x for wall in walls])
    max_x = max([wall.max_x for wall in walls])
    min_y = min([wall.min_y for wall in walls])
    max_y = max([wall.max_y for wall in walls])
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    for wall in walls:
        left_bottom = wall.endpoint3
        right_top = wall.endpoint1
        width = right_top[0] - left_bottom[0]
        height = right_top[1] - left_bottom[1]    
        rect = Rectangle(left_bottom, width, height,
                         linewidth=1, edgecolor='k', facecolor='k')
        ax.add_patch(rect)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    if axis_off:
        ax.axis('off')
    ax.set_aspect('equal')