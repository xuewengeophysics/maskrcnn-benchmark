import torch
import numpy as np

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class Keypoints(object):
    def __init__(self, keypoints, size, mode=None):
        # FIXME remove check once we have better integration with device
        # in my version this would consistently return a CPU tensor
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device('cpu')
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        num_keypoints = keypoints.shape[0]
        if num_keypoints:
            keypoints = keypoints.view(num_keypoints, -1, 3)
        
        # TODO should I split them?
        # self.visibility = keypoints[..., 2]
        self.keypoints = keypoints# [..., :2]

        self.size = size
        self.mode = mode
        self.extra_fields = {}

    def crop(self, box):
        raise NotImplementedError()

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_w, ratio_h = ratios
        resized_data = self.keypoints.clone()
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
        keypoints = type(self)(resized_data, size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                    "Only FLIP_LEFT_RIGHT implemented")

        flip_inds = type(self).FLIP_INDS
        flipped_data = self.keypoints[:, flip_inds]
        width = self.size[0]
        TO_REMOVE = 1
        # Flip x coordinates
        flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE

        # Maintain COCO convention that if visibility == 0, then x, y = 0
        inds = flipped_data[..., 2] == 0
        flipped_data[inds] = 0

        keypoints = type(self)(flipped_data, self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def to(self, *args, **kwargs):
        keypoints = type(self)(self.keypoints.to(*args, **kwargs), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            keypoints.add_field(k, v)
        return keypoints

    def __getitem__(self, item):
        keypoints = type(self)(self.keypoints[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v[item])
        return keypoints

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.keypoints))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s


def _create_flip_indices(names, flip_map):
    full_flip_map = flip_map.copy()
    full_flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in full_flip_map else full_flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return torch.tensor(flip_indices)


class PersonKeypoints(Keypoints):
    NAMES = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    FLIP_MAP = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }


# TODO this doesn't look great
PersonKeypoints.FLIP_INDS = _create_flip_indices(PersonKeypoints.NAMES, PersonKeypoints.FLIP_MAP)
def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines
PersonKeypoints.CONNECTIONS = kp_connections(PersonKeypoints.NAMES)


# TODO make this nicer, this is a direct translation from C2 (but removing the inner loop)
def keypoints_to_heatmap(keypoints, rois, heatmap_size):
    """
    input:
        keypoint的维度为[person_num, 17, 3], "3"代表(x, y, visibility)
        rois的维度为[person_num, 4], "4"代表(x1, y1, x2, y2)
        heatmap_size的大小等于cfg.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION, 配置文件中为56
    output:
        heatmaps的维度为[person_num, 17], 代表关键点的ground truth特征
        valid的维度为[person_num, 17], 代表关键点是否存在
    """
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()
    
    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    ##heatmaps的维度为[person_num, 17], valid的维度为[person_num, 17]
    import ipdb;ipdb.set_trace()
    return heatmaps, valid



# Generate the target heatmap using gaussian
def keypoints_to_heatmap_gaussian(keypoints, rois, heatmap_size):
    """
    input:
        keypoint的维度为[person_num, 17, 3], "3"代表(x, y, visibility)
        rois的维度为[person_num, 4], "4"代表(x1, y1, x2, y2)
        heatmap_size的大小等于cfg.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION, 配置文件中为56
    output:
        heatmaps的维度为[person_num, 17, heatmap_size, heatmap_size], 代表关键点的ground truth特征
        valid的维度为[person_num, 17], 代表关键点是否存在
    """
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()
    
    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    # heatmaps = lin_ind * valid

    ##heatmaps的维度为[person_num, 17, heatmap_size, heatmap_size], valid的维度为[person_num, 17]
    heatmaps = []
    for k in range(len(keypoints)):
        # print("ann keypoints = ", ann['keypoints'])
        # ipdb.set_trace()
        joints_3d =np.array(keypoints[k].cpu().tolist())  ##17代表关键点个数，3代表(x, y, v)，v为{0:不存在, 1:存在但不可见, 2:存在且可见}
        x1, y1, x2, y2 = list(map(int, np.array(rois[k].cpu().tolist())))
        w = x2 - x1
        h = y2 - y1
        enlarge_ratio_h, enlarge_ratio_w = heatmap_size/h, heatmap_size/w
        
        nonzero_x = joints_3d[..., 0].nonzero()[0]  ##array，x坐标不为0的关键点的index组成的array
        nonzero_y = joints_3d[..., 1].nonzero()[0]  ##array，y坐标不为0的关键点的index组成的array

        # import ipdb;ipdb.set_trace()
        joints_3d[..., 0][nonzero_x] = (joints_3d[..., 0][nonzero_x] - x1) * enlarge_ratio_w  ##各个关键点在大小为[WIDTH, HEIGHT]的resize图上的位置的x坐标
        joints_3d[..., 1][nonzero_y] = (joints_3d[..., 1][nonzero_y] - y1) * enlarge_ratio_h  ##各个关键点在大小为[WIDTH, HEIGHT]的resize图上的位置的y坐标

        # cv2.imwrite("./croped_person{}.png".format(k), croped_person)
        joints_3d_visible = np.zeros((17, 3))
        for i, kps in enumerate(joints_3d):
            v = kps[-1]
            if v > 0:
                joints_3d_visible[i] = np.array([1,1,1])
        WIDTH = heatmap_size
        HEIGHT = heatmap_size
        heatmap_w = heatmap_size
        heatmap_h = heatmap_size
        cfg = {'image_size': np.array([WIDTH, HEIGHT]), 'num_joints': 17, 'heatmap_size': np.array([heatmap_w, heatmap_h])}
        #  result = generate_guassian_heatmap(cfg, joints_3d, joints_3d_visible)
        targets, targets_visible = generate_target(cfg, joints_3d, joints_3d_visible) # 包含了17个heatmap

        heatmaps.append(targets)
    heatmaps = torch.from_numpy(np.array(heatmaps)).cuda()
    # import ipdb;ipdb.set_trace()

    return heatmaps, valid


def generate_target(cfg, joints_3d, joints_3d_visible, sigma=3):
    """Generate the target heatmap.

    Args:
        cfg (dict): data config
        joints_3d: np.ndarray ([num_joints, 3])
        joints_3d_visible: np.ndarray ([num_joints, 3])

    Returns:
        tuple: A tuple containing targets.

        - target: Target heatmaps.
        - target_weight: (1: visible, 0: invisible)
    ??? image_size 在其中的作用, 表明关键点坐标的相对位置
    """
    num_joints = cfg['num_joints']
    image_size = cfg['image_size']
    heatmap_size = cfg['heatmap_size']
    target_weight = np.zeros((num_joints, 1), dtype=np.float32)
    target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                        dtype=np.float32)
    tmp_size = sigma * 3
    for joint_id in range(num_joints):
        heatmap_vis = joints_3d_visible[joint_id, 0]
        target_weight[joint_id] = heatmap_vis
        feat_stride = image_size / heatmap_size
        mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[
                0] < 0 or br[1] < 0:
            # print("warn: {}".format(joint_id))
            target_weight[joint_id] = 0
        if target_weight[joint_id] > 0.5:
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]
            x0 = y0 = size // 2
            # The gaussian is not normalized,
            # we want the center value to equal 1
            g = np.exp(-((x - x0)**2 + (y - y0)**2) /
                        (2 * sigma**2))
            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return target, target_weight