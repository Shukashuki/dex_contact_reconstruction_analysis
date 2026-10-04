"""Independent C2Dex Eq. 6--11 trajectory retargeting (residual PPO excluded).

Export MuJoCo constants with the project venv, then optimize with the existing
torch environment. The paper does not release finger-vertex assignments,
contact offsets, edge-weight kernel or precise collision/smoothness definitions.
These choices are explicit in the report. Convex-link plane fields approximate
link SDFs; the object uses the reconstruction module's sampled SDF grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')


def export_model(output: Path, side: str, robot_xml: Path, robot_contract: Path):
    from retarget.export_model import export
    export(output, robot_xml, robot_contract)


class RobotFK:
    def __init__(self, data, device):
        import torch
        self.torch = torch
        self.device = device
        self.parent = data['parent'].astype(int)
        self.joint = data['body_joint'].astype(int)
        self.pos = self.tensor(data['pos'])
        self.rot = self.tensor(Rotation.from_quat(np.roll(data['quat'], -1, axis=1)).as_matrix())
        self.axis = self.tensor(data['axis'])
        self.anchor = self.tensor(data['anchor'])

    def tensor(self, array):
        return self.torch.as_tensor(array, dtype=self.torch.float32, device=self.device)

    def __call__(self, q):
        from perception.mano_torch import axis_angle_to_matrix
        B = len(q)
        rotations = [self.rot[0].expand(B, 3, 3)]
        positions = [self.pos[0].expand(B, 3)]
        for b in range(1, len(self.parent)):
            p = self.parent[b]
            local_R = self.rot[b].expand(B, 3, 3)
            local_p = self.pos[b].expand(B, 3)
            j = self.joint[b]
            if j >= 0:
                joint_R = axis_angle_to_matrix(q[:, j, None] * self.axis[j])
                local_p = local_p + (local_R @ (self.anchor[j] - (joint_R @ self.anchor[j, :, None]).squeeze(-1))[:, :, None]).squeeze(-1)
                local_R = local_R @ joint_R
            positions.append(positions[p] + (rotations[p] @ local_p[:, :, None]).squeeze(-1))
            rotations.append(rotations[p] @ local_R)
        return self.torch.stack(rotations, 1), self.torch.stack(positions, 1)


def optimize(args):
    import torch
    from scipy.spatial import ConvexHull, Delaunay
    from scipy.spatial.distance import cdist
    from perception.mano_torch import ManoTorch, axis_angle_to_matrix, load_mano_npz
    from perception.c2dex_optimize import sdf_grid, sample_sdf, _rotation_matrices
    from perception.contact_sdf import finger_vertex_groups
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    constants = np.load(args.model, allow_pickle=True)
    fk = RobotFK(constants, device)
    tensor = fk.tensor
    with torch.no_grad():
        check = fk(tensor(constants['test_q']))[1].cpu().numpy()
    fk_error = float(np.max(np.abs(check - constants['test_positions'])))
    if fk_error > 1e-6:
        raise ValueError(f'FK disagrees with MuJoCo by {fk_error}m')
    human = np.load(args.human, allow_pickle=True)
    contacts = np.load(args.contacts, allow_pickle=True)
    template = np.load(args.template, allow_pickle=True)
    T = len(human['mano_pose'])
    mano_np = load_mano_npz(str(human['side']))
    mano = ManoTorch(mano_np, device)
    human_R = _rotation_matrices(human['wrist_pose'], device)
    obj_R = _rotation_matrices(human['obj_pose'], device)
    obj_t = tensor(human['obj_pose'][:, :3])
    with torch.no_grad():
        hj, hv = mano(torch.cat([torch.zeros(T, 3, device=device), tensor(human['mano_pose'])], -1),
                      tensor(human['mano_betas']), torch.zeros(T, 3, device=device))
        mano_scale = float(human['mano_scale']) if 'mano_scale' in human.files else 1.0
        hv = hj[:, :1] + mano_scale * (hv - hj[:, :1])
        hj = hj[:, :1] + mano_scale * (hj - hj[:, :1])
        # MANO's internal 16-joint order -> the established 21-joint convention.
        parts = [hj[:, :1]]
        for ids, tip in zip(((13,14,15),(1,2,3),(4,5,6),(10,11,12),(7,8,9)), (745,317,444,556,673)):
            parts.extend([hj[:, ids], hv[:, tip:tip+1]])
        hj_world = torch.cat(parts, 1) - hj[:, :1]
        hj_world = (human_R[:, None] @ hj_world[:, :, :, None]).squeeze(-1) + tensor(human['wrist_pose'][:, None, :3])
        hv_world = (human_R[:, None] @ (hv - hj[:, :1])[:, :, :, None]).squeeze(-1) + tensor(human['wrist_pose'][:, None, :3])
    node_bodies = constants['nodes'].astype(int)
    target_nodes = hj_world[:, constants['human_ids'].astype(int)]
    if args.clip:
        import h5py
        with h5py.File(args.clip / 'joints.hdf5', 'r') as handle:
            gt_joints = handle[f'{str(human["side"])}_hand'][:]
        if len(gt_joints) != T:
            raise ValueError('GT joints and human trajectory length differ')
        target_nodes = tensor(gt_joints[:, constants['human_ids'].astype(int)])
    C = tensor(constants['C'])
    robot_R0 = human_R
    robot_p0 = tensor(human['wrist_pose'][:, :3])
    if args.initial_robot_wrist:
        robot_pose = np.load(args.initial_robot_wrist)['wrist_pose_W']
        if robot_pose.shape != (T, 7):
            raise ValueError('initial robot wrist must match human trajectory')
        robot_R0 = _rotation_matrices(robot_pose, device)
        robot_p0 = tensor(robot_pose[:, :3])
        C = tensor(np.eye(3))
    q = tensor(np.broadcast_to(constants['limits'].mean(1), (T, len(constants['limits']))).copy()).requires_grad_(True)
    p = robot_p0.clone().requires_grad_(True)
    dr = torch.zeros(T, 3, device=device, requires_grad=True)
    limits = tensor(constants['limits'])

    def world():
        local_R, local_p = fk(q)
        wrist_R = axis_angle_to_matrix(dr) @ robot_R0
        base_R = wrist_R @ C
        body_R = base_R[:, None] @ local_R
        body_p = (base_R[:, None] @ local_p[:, :, :, None]).squeeze(-1) + p[:, None]
        return wrist_R, body_R, body_p

    # Independent keypoint initializer; no topo solution used as optimization seed.
    initializer = torch.optim.Adam([q], lr=.02)
    for _ in range(500):
        initializer.zero_grad()
        node_p = world()[2][:, node_bodies]
        loss = ((node_p - target_nodes) ** 2).sum(-1).mean()
        loss.backward()
        initializer.step()
        with torch.no_grad():
            q.copy_(torch.maximum(torch.minimum(q, limits[:, 1]), limits[:, 0]))
    q_initial = q.detach().clone()
    print('keypoint initializer complete', flush=True)
    initial_R, initial_body_R, initial_body_p = [x.detach() for x in world()]
    import trimesh
    mesh = trimesh.load(args.mesh, force='mesh', process=False)
    vertices = np.asarray(mesh.vertices) * (.01 if args.mesh.name.endswith('_cm.obj') else 1.)
    # Fixed canonical object points, deterministic farthest-point selection.
    sampled = [int(np.argmax(np.linalg.norm(vertices - vertices.mean(0), axis=1)))]
    distances = np.linalg.norm(vertices - vertices[sampled[0]], axis=1)
    for _ in range(39):
        sampled.append(int(np.argmax(distances)))
        distances = np.minimum(distances, np.linalg.norm(vertices - vertices[sampled[-1]], axis=1))
    object_nodes = (obj_R[:, None] @ tensor(vertices[sampled])[None, :, :, None]).squeeze(-1) + obj_t[:, None]
    reference = torch.cat([target_nodes, object_nodes], 1).cpu().numpy()
    graph = []
    for points in reference:
        adjacency = np.zeros((len(points), len(points)), bool)
        for simplex in Delaunay(points).simplices:
            adjacency[np.ix_(simplex, simplex)] = True
        np.fill_diagonal(adjacency, False)
        distance = cdist(points, points)
        sigma = np.median(distance[adjacency])
        weights = np.exp(-distance / max(sigma, 1e-9)) * adjacency
        weights /= np.maximum(weights.sum(1, keepdims=True), 1e-12)
        graph.append(np.eye(len(points)) - weights)
    L = tensor(np.stack(graph))
    lap_target = L @ tensor(reference)

    # A single fixed representative human vertex per finger (Eq. 6).
    # Choose the vertex present in the largest number of frame-expanded stable
    # contacts. Map it to the nearest same-finger robot surface sample once at
    # the first active frame. Both choices are held fixed for the trajectory.
    groups = finger_vertex_groups(mano_np.weights)
    segment = contacts['segment_id'].astype(int)
    sv = contacts['stable_vertex'].astype(int)
    ss = contacts['stable_segment'].astype(int)
    sp = contacts['stable_point_object']
    surface = constants['surface']
    sb = constants['surface_body'].astype(int)
    body_names = constants['body_names']
    mapping = []
    target_contact = np.zeros((T, 5, 3))
    active = np.zeros((T, 5), bool)
    offsets = np.zeros((5, 3))
    contact_bodies = np.ones(5, int)
    for f, name in enumerate(FINGERS):
        candidates = sorted(set(sv) & set(groups[name]))
        if not candidates:
            mapping.append({'finger': name, 'vertex': None})
            continue
        representative = max(candidates, key=lambda v: sum(np.count_nonzero(segment == s) for s in ss[sv == v]))
        for s, v, point in zip(ss, sv, sp):
            if v == representative:
                active[segment == s, f] = True
                target_contact[segment == s, f] = point
        t = int(np.where(active[:, f])[0][0])
        select = np.asarray([name in str(body_names[b]) for b in sb])
        ids = np.where(select)[0]
        points = (initial_body_R[t, sb[ids]] @ tensor(surface[ids])[:, :, None]).squeeze(-1) + initial_body_p[t, sb[ids]]
        nearest = ids[int(torch.argmin(((points - hv_world[t, representative]) ** 2).sum(-1)))]
        offsets[f], contact_bodies[f] = surface[nearest], sb[nearest]
        mapping.append({'finger': name, 'vertex': int(representative), 'robot_body': str(body_names[sb[nearest]]),
                        'local_surface_point': offsets[f].tolist(), 'active_frames': int(active[:, f].sum())})
    active_t = tensor(active)
    target_contact_world = (obj_R[:, None] @ tensor(target_contact)[:, :, :, None]).squeeze(-1) + obj_t[:, None]
    auxiliary = args.output.parent / 'debug'
    auxiliary.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(auxiliary / (args.output.stem + '.targets.npz'), active=active,
                        target_contact_object=target_contact, representative_vertices=[m['vertex'] if m['vertex'] is not None else -1 for m in mapping])

    # Convex-hull link SDF approximation: max outward plane distance. A negative
    # value is penetration into a hull; positive values underestimate edge/corner
    # distances. Sample original collision mesh surfaces for object/self tests.
    unique_bodies = sorted(set(sb))
    body_samples, sample_bodies, hulls = [], [], {}
    for b in unique_bodies:
        points = np.unique(surface[sb == b], axis=0)
        # Bounded convex field size keeps the all-frame autograd problem within
        # an 8GB GPU. Farthest-point supports define an explicit approximation.
        support = [int(np.argmax(np.linalg.norm(points-points.mean(0), axis=1)))]
        dist = np.linalg.norm(points-points[support[0]], axis=1)
        for _ in range(min(64, len(points))-1):
            support.append(int(np.argmax(dist)))
            dist = np.minimum(dist, np.linalg.norm(points-points[support[-1]], axis=1))
        hull = ConvexHull(points[support])
        hulls[b] = hull.equations
        indices = np.linspace(0, len(points)-1, min(16, len(points))).astype(int)
        body_samples.extend(points[indices])
        sample_bodies.extend([b] * len(indices))
    sampled_surface = tensor(np.asarray(body_samples))
    sample_bodies = np.asarray(sample_bodies)
    # Cross-finger pairs and finger-palm pairs, excluding rigid/adjacent links.
    pairs = []
    def digit(b):
        return next((f for f in FINGERS if f in str(body_names[b])), 'palm')
    for a in unique_bodies:
        for b in unique_bodies:
            if a == b or digit(a) == digit(b):
                continue
            if constants['parent'][a] == b or constants['parent'][b] == a:
                continue
            pairs.append((a, b))
    nplanes = max(len(hulls[b]) for _, b in pairs)
    planes = np.zeros((len(pairs), nplanes, 4))
    planes[:, :, 3] = -1e3
    for k, (_, b) in enumerate(pairs):
        planes[k, :len(hulls[b])] = hulls[b]
    source_samples = np.stack([surface[sb == a][np.linspace(0, np.count_nonzero(sb == a)-1, 8).astype(int)] for a, _ in pairs])
    pair_a, pair_b = np.asarray(pairs).T
    pair_points, plane = tensor(source_samples), tensor(planes)
    grid, lo, hi = sdf_grid(args.mesh, args.sdf_resolution, .03)
    print('interaction graphs and collision fields ready', flush=True)
    sdf = tensor(grid).reshape(1, 1, *grid.shape)
    lo, hi = tensor(lo), tensor(hi)
    from scipy.spatial import cKDTree
    object_surface_normals = np.asarray(mesh.vertex_normals)
    normal_ids = cKDTree(vertices).query(target_contact.reshape(-1, 3))[1]
    contact_target_normals = tensor(object_surface_normals[normal_ids].reshape(T, 5, 3))
    robot_normals = np.zeros((5, 3))
    for f in range(5):
        if active[:, f].any():
            equations = hulls[contact_bodies[f]]
            robot_normals[f] = equations[np.argmax(equations[:, :3] @ offsets[f] + equations[:, 3]), :3]

    def losses():
        wrist_R, body_R, body_p = world()
        points = (body_R[:, sample_bodies] @ sampled_surface[None, :, :, None]).squeeze(-1) + body_p[:, sample_bodies]
        nodes = body_p[:, node_bodies]
        all_nodes = torch.cat([nodes, object_nodes], 1)
        lap = ((L @ all_nodes - lap_target) ** 2).sum((1, 2)).mean()
        contacts_world = (body_R[:, contact_bodies] @ tensor(offsets)[None, :, :, None]).squeeze(-1) + body_p[:, contact_bodies]
        contact = ((((contacts_world - target_contact_world) ** 2).sum(-1) * active_t).sum(1) / active_t.sum(1).clamp(min=1)).mean()
        canonical = (obj_R[:, None].transpose(-1, -2) @ (points - obj_t[:, None])[:, :, :, None]).squeeze(-1)
        depth = torch.relu(-sample_sdf(sdf, canonical, lo, hi))
        # Transform samples on link a into link b's local coordinates.
        pair_world = (body_R[:, pair_a, None] @ pair_points[None, :, :, :, None]).squeeze(-1) + body_p[:, pair_a, None]
        pair_local = (body_R[:, pair_b, None].transpose(-1, -2) @ (pair_world-body_p[:, pair_b, None])[:, :, :, :, None]).squeeze(-1)
        plane_d = torch.einsum('tpni,pfi->tpnf', pair_local, plane[:, :, :3]) + plane[None, :, None, :, 3]
        self_depth = torch.relu(-plane_d.max(-1).values)
        pene = (depth ** 2).mean() + (self_depth ** 2).mean()
        smooth = ((q[1:] - q[:-1]) ** 2).sum(-1).mean() + ((p[1:] - p[:-1]) ** 2).sum(-1).mean()
        smooth = smooth + ((wrist_R[1:] - wrist_R[:-1]) ** 2).sum((1, 2)).mean()
        total = 500*lap + 2e4*contact + 1e5*pene + smooth
        active_distance = torch.linalg.vector_norm(contacts_world-target_contact_world, dim=-1)[active]
        contact_normals = (body_R[:, contact_bodies] @ tensor(robot_normals)[None, :, :, None]).squeeze(-1)
        target_normals_world = (obj_R[:, None] @ contact_target_normals[:, :, :, None]).squeeze(-1)
        angles = torch.acos((-(contact_normals*target_normals_world).sum(-1)).clamp(-1, 1))[active]
        return total, lap, contact, pene, smooth, depth.max(), self_depth.max(), active_distance.mean(), angles.mean()

    def metrics(vals):
        return {'loss': float(vals[0]), 'laplacian': float(vals[1]), 'contact_rms_mm': float(torch.sqrt(vals[2])*1000),
                'penetration_loss': float(vals[3]), 'smoothness': float(vals[4]),
                'object_sample_max_penetration_mm': float(vals[5]*1000), 'self_sample_max_penetration_mm': float(vals[6]*1000),
                'E_prec_representative_mean_mm': float(vals[7]*1000),
                'E_align_opposed_normals_mean_deg': float(vals[8]*180/np.pi)}
    with torch.no_grad():
        initial = metrics(losses())
    optimizer = torch.optim.Adam([q, p, dr], lr=.02)
    history = []
    for step in range(args.steps):
        optimizer.zero_grad()
        vals = losses()
        if not torch.isfinite(vals[0]):
            raise ValueError(f'nonfinite loss at {step}')
        vals[0].backward()
        optimizer.step()
        with torch.no_grad():
            q.copy_(torch.maximum(torch.minimum(q, limits[:, 1]), limits[:, 0]))
        if step == 0 or (step+1) % 100 == 0:
            item = {'step': step+1, **metrics(vals)}
            history.append(item)
            print(json.dumps(item), flush=True)
    with torch.no_grad():
        final = metrics(losses())
        wrist_R = world()[0].cpu().numpy()
    evaluated = {}
    saved = [x.detach().clone() for x in (q, p, dr)]
    for path in args.evaluate_trajectories:
        other = np.load(path, allow_pickle=True)
        other_R = _rotation_matrices(other['wrist_pose_W'], device)
        relative = (other_R @ robot_R0.transpose(-1, -2)).cpu().numpy()
        with torch.no_grad():
            q.copy_(tensor(other['hand_qpos']))
            p.copy_(tensor(other['wrist_pose_W'][:, :3]))
            dr.copy_(tensor(Rotation.from_matrix(relative).as_rotvec()))
            evaluated[str(path)] = metrics(losses())
    with torch.no_grad():
        for variable, value in zip((q, p, dr), saved):
            variable.copy_(value)
    quaternion = Rotation.from_matrix(wrist_R).as_quat()
    wrist = np.column_stack([p.detach().cpu().numpy(), quaternion[:, 3], quaternion[:, :3]])
    payload = {k: template[k] for k in template.files}
    payload['hand_qpos'] = q.detach().cpu().numpy().astype(np.float32)
    payload['wrist_pose_W'] = wrist.astype(np.float32)
    payload['source'] = np.asarray(str(template['source']).split('|')[0] + '|c2dex_ret_v1')
    np.savez_compressed(args.output, **payload)
    initial_payload = dict(payload)
    initial_payload['hand_qpos'] = q_initial.cpu().numpy().astype(np.float32)
    initial_payload['wrist_pose_W'] = (robot_pose if args.initial_robot_wrist else human['wrist_pose'])
    initial_payload['source'] = np.asarray(str(template['source']).split('|')[0] + '|c2dex_keypoint_initializer')
    np.savez_compressed(args.output.with_suffix('.initializer.npz'), **initial_payload)
    report = {'status': ('kinematic_trajectory_optimization_complete_residual_ppo_not_run'
                         if args.steps else 'kinematic_evaluation_complete'),
              'paper': 'https://arxiv.org/html/2608.07045', 'implementation': 'independent_reproduction',
              'human': str(args.human), 'contacts': str(args.contacts), 'output': str(args.output),
              'keypoint_source': str(args.clip / 'joints.hdf5') if args.clip else 'input MANO trajectory',
              'robot_wrist_frame': 'handroot' if args.initial_robot_wrist else 'MANO wrist mapped by C_side',
              'initial_robot_wrist': str(args.initial_robot_wrist) if args.initial_robot_wrist else None,
              'embodiment': 'revo3', 'paper_embodiment': 'Inspire', 'steps': args.steps, 'learning_rate': .02,
              'weights': {'laplacian':500, 'contact':20000, 'penetration':100000, 'smoothness':1},
              'device': str(device), 'fk_mujoco_max_error_m': fk_error, 'mapping': mapping,
              'initial': initial, 'final': final, 'history': history,
              'evaluated_trajectories': evaluated,
              'approximations': ['fixed representative vertex chosen by stable-contact duration per finger',
                                 'robot contact offset chosen from same-finger surface at first active frame',
                                 'distance-weight kernel exp(-distance/median edge length)',
                                 'object vertex-distance SDF grid and 16 collision mesh samples per link',
                                 'self collision uses 64-support convex-hull plane fields with cross-finger/finger-palm samples',
                                 'squared penetration and first-difference joint/translation/rotation smoothness',
                                 'keypoint Adam initializer: 500 steps, fixed human wrist',
                                 'no ManipTrans residual PPO; not full end-to-end C2Dex']}
    args.output.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'history'}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export-model', type=Path)
    p.add_argument('--robot-xml', type=Path)
    p.add_argument('--robot-contract', type=Path)
    p.add_argument('--side', default='right')
    p.add_argument('--model', type=Path)
    p.add_argument('--human', type=Path)
    p.add_argument('--contacts', type=Path)
    p.add_argument('--template', type=Path)
    p.add_argument('--mesh', type=Path)
    p.add_argument('--clip', type=Path, help='use GT keypoints, matching the paper retarget-isolation protocol')
    p.add_argument('--initial-robot-wrist', type=Path, help='archive handroot poses; optimizes and exports in the handroot frame')
    p.add_argument('--output', type=Path)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--sdf-resolution', type=int, default=48)
    p.add_argument('--evaluate-trajectories', type=Path, nargs='*', default=[])
    args = p.parse_args()
    if args.export_model:
        if args.robot_xml is None or args.robot_contract is None:
            p.error('--export-model requires --robot-xml and --robot-contract')
        export_model(args.export_model, args.side, args.robot_xml, args.robot_contract)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        optimize(args)


if __name__ == '__main__':
    main()
