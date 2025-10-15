"""  
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License 
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
 
import sys
import math
import re

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return (w, x, y, z)

def q_conjugate(q):
    return (q[0], -q[1], -q[2], -q[3])

def qv_mult(q, v):
    qv = (0.0, v[0], v[1], v[2])
    res = q_mult(q_mult(q, qv), q_conjugate(q))
    return [res[1], res[2], res[3]]

def quat_to_mat(q):
    # q = (w, x, y, z)
    x, y, z, w = q[1], q[2], q[3], q[0]
    xx = x * x; yy = y * y; zz = z * z
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z
    return [
        1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy), 0,
        2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx), 0,
        2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy), 0,
        0, 0, 0, 1
    ]

def mat_to_quat(m):
    # m is 16-element list (4x4), extract 3x3 rotation part
    # Assuming no scale or shear, pure rotation
    t = m[0] + m[5] + m[10]
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[9] - m[6]) * s
        y = (m[2] - m[8]) * s
        z = (m[4] - m[1]) * s
    elif m[0] > m[5] and m[0] > m[10]:
        s = 0.5 / math.sqrt(1.0 + m[0] - m[5] - m[10])
        x = 0.25 / s
        y = (m[1] + m[4]) * s
        z = (m[2] + m[8]) * s
        w = (m[9] - m[6]) * s
    elif m[5] > m[10]:
        s = 0.5 / math.sqrt(1.0 + m[5] - m[0] - m[10])
        y = 0.25 / s
        x = (m[1] + m[4]) * s
        z = (m[6] + m[9]) * s
        w = (m[2] - m[8]) * s
    else:
        s = 0.5 / math.sqrt(1.0 + m[10] - m[0] - m[5])
        z = 0.25 / s
        x = (m[2] + m[8]) * s
        y = (m[6] + m[9]) * s
        w = (m[4] - m[1]) * s
    q = (w, x, y, z)
    return normalize_quat(q)

def diag_scale_mat(scale):
    return [
        scale[0], 0, 0, 0,
        0, scale[1], 0, 0,
        0, 0, scale[2], 0,
        0, 0, 0, 1
    ]

def translation_mat(trans):
    return [
        1, 0, 0, trans[0],
        0, 1, 0, trans[1],
        0, 0, 1, trans[2],
        0, 0, 0, 1
    ]

def mat_mult(m1, m2):
    res = [0.0] * 16
    for i in range(4):
        for j in range(4):
            for k in range(4):
                res[i*4 + j] += m1[i*4 + k] * m2[k*4 + j]
    return res

def mat_mult_vec(mat, v):
    vh = v + [1.0]
    res = [0.0] * 3
    for j in range(3):
        for k in range(4):
            res[j] += mat[j*4 + k] * vh[k]
    return res

def mat4_inv(mat):
    trans = [mat[3], mat[7], mat[11]]
    col0 = [mat[0], mat[4], mat[8]]
    col1 = [mat[1], mat[5], mat[9]]
    col2 = [mat[2], mat[6], mat[10]]
    s0 = math.sqrt(sum(c**2 for c in col0))
    s1 = math.sqrt(sum(c**2 for c in col1))
    s2 = math.sqrt(sum(c**2 for c in col2))
    if s0 == 0 or s1 == 0 or s2 == 0: 
        return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
    scale = [s0, s1, s2]
    rot_col0 = [c / s0 for c in col0]
    rot_col1 = [c / s1 for c in col1]
    rot_col2 = [c / s2 for c in col2]
    inv_rot = [
        rot_col0[0], rot_col1[0], rot_col2[0], 0,
        rot_col0[1], rot_col1[1], rot_col2[1], 0,
        rot_col0[2], rot_col1[2], rot_col2[2], 0,
        0, 0, 0, 1
    ]
    inv_scale_mat = diag_scale_mat([1/s for s in scale])
    inv_trans_mat = translation_mat([-trans[0], -trans[1], -trans[2]])
    inv_mat = mat_mult(inv_scale_mat, mat_mult(inv_rot, inv_trans_mat))
    return inv_mat

def parse_float_list(s):
    try:
        return [float(x.strip()) for x in s.strip('() ').split() if x.strip()]
    except:
        return []

def parse_vec3(lst):
    return [lst[j] if j < len(lst) else 0.0 for j in range(3)]

def parse_quat_raw(lst):
    if len(lst) < 4:
        return (1.0, 0.0, 0.0, 0.0)
    x, y, z, w = lst
    return (w, x, y, z)

def normalize_quat(q):
    norm = math.sqrt(sum(c**2 for c in q))
    if norm == 0:
        return (0.0, 0.0, 0.0, 0.0)
    q_norm = tuple(c / norm for c in q)
    if q_norm[0] < 0.0:
        q_norm = tuple(-c for c in q_norm)
    return q_norm

def parse_joint_line(line):
    tokens = line.split()
    if len(tokens) < 2: return None
    name = tokens[0].strip('"')
    parent_idx = int(tokens[1])
    floats = []
    for t in tokens[2:]:
        if t == '(' or t == ')': continue
        try:
            floats.append(float(t))
        except ValueError:
            pass
    if len(floats) < 18: return None
    quat_raw = floats[8:12]
    scale = parse_vec3(floats[12:15])
    pos = parse_vec3(floats[15:18])
    quat = normalize_quat(parse_quat_raw(quat_raw))
    return {'name': name, 'parent': parent_idx, 'pos': pos, 'quat': quat, 'scale': scale}

def main(input_file, output_file):
    with open(input_file, 'r') as f:
        text = f.read()
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    i = 0
    # Parse init: collect until "}"
    while i < len(lines) and "init {" not in lines[i]: i += 1
    i += 1
    init_lines = []
    while i < len(lines) and lines[i] != "}":
        init_lines.append(lines[i])
        i += 1
    i += 1  # Skip }
    num_joints = 0
    num_meshes = 0
    for line in init_lines:
        parts = line.split()
        if len(parts) >= 2:
            if parts[0] == "numJoints": num_joints = int(parts[1])
            elif parts[0] == "numMeshes": num_meshes = int(parts[1])

    # Parse joints
    joints = []
    while i < len(lines) and "joints {" not in lines[i]: i += 1
    i += 1
    for _ in range(num_joints):
        while i < len(lines) and not lines[i].startswith('"'): i += 1
        if i >= len(lines): break
        joint_data = parse_joint_line(lines[i])
        if joint_data:
            joints.append(joint_data)
        else:
            print(f"Warning: Skipping invalid joint line: {lines[i]}")
        i += 1
    while i < len(lines) and lines[i] != "}": i += 1
    i += 1  # Skip }

    # Compute bind_mats hierarchically (absolute transforms)
    identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    bind_mats = [identity[:] for _ in joints]
    for idx in range(len(joints)):
        parent = joints[idx]['parent']
        parent_mat = bind_mats[parent] if parent >= 0 else identity
        rot_mat = quat_to_mat(joints[idx]['quat'])
        scale_mat = diag_scale_mat(joints[idx]['scale'])
        trans_mat = translation_mat(joints[idx]['pos'])
        local_mat = mat_mult(mat_mult(trans_mat, rot_mat), scale_mat)  # T * R * S
        bind_mats[idx] = mat_mult(parent_mat, local_mat)

    # Output header and joints with absolute pos and quat
    with open(output_file, 'w') as out:
        out.write("MD5Version 10\n")
        out.write('commandline "converted from md6mesh"\n')
        out.write(f"numJoints {len(joints)}\n")
        out.write(f"numMeshes {num_meshes}\n")
        out.write("\n")
        out.write("joints {\n")
        for idx, j in enumerate(joints):
            absolute_pos = (bind_mats[idx][3], bind_mats[idx][7], bind_mats[idx][11])
            absolute_quat = mat_to_quat(bind_mats[idx])
            qx, qy, qz = absolute_quat[1], absolute_quat[2], absolute_quat[3]
            out.write(f'  "{j["name"]}" {j["parent"]} ( {absolute_pos[0]:.6f} {absolute_pos[1]:.6f} {absolute_pos[2]:.6f} ) ( {qx:.6f} {qy:.6f} {qz:.6f} )\n')
        out.write("}\n\n")

        # Parse and output meshes
        for m in range(num_meshes):
            while i < len(lines) and "mesh {" not in lines[i]: i += 1
            if i >= len(lines): break
            i += 1  # Enter mesh {
            shader = ""
            # Parse header
            while i < len(lines) and "verts " not in lines[i]:
                if 'shader "' in lines[i]:
                    shader = lines[i].split('"')[1]
                i += 1
            if i >= len(lines): continue
            # Parse verts N
            parts = lines[i].split()
            orig_vert_count = int(parts[1]) if len(parts) > 1 else 0
            i += 1  # Enter verts {

            # Parse verts (use dict for out-of-order)
            mesh_poses = {}
            mesh_uvs = {}
            mesh_vert_influences = {}
            max_vert_idx = -1
            while i < len(lines) and "}" not in lines[i]:
                line = lines[i]
                idx_match = re.search(r'vert\s+(\d+)', line)
                if not idx_match:
                    i += 1
                    continue
                vert_idx = int(idx_match.group(1))
                match = re.search(r'\(\s*([^)]*)\)\s*\(\s*([^)]*)\)\s*\(\s*([^)]*)\)', line)
                if not match:
                    print(f"Warning: Skipping invalid vert line: {line}")
                    i += 1
                    continue
                pos_str, uv_str, wt_str = [s.strip() for s in match.groups()]
                pos_list = parse_float_list(pos_str)
                if len(pos_list) < 3:
                    print(f"Warning: Invalid pos for vert {vert_idx}")
                    i += 1
                    continue
                pos = pos_list[:3]
                uv_list = parse_float_list(uv_str)
                u = uv_list[0] if uv_list else 0.0
                v_uv = uv_list[1] if len(uv_list) > 1 else 0.0  # No flip
                wt_parts = parse_float_list(wt_str)
                if len(wt_parts) < 4:
                    print(f"Warning: Invalid weights for vert {vert_idx}")
                    i += 1
                    continue
                bone_ids = [int(wt_parts[k]) if k < len(wt_parts) and wt_parts[k] >= 0 else -1 for k in range(4)]
                raw_wts = [wt_parts[k] if k < len(wt_parts) else 0.0 for k in range(4, 8)]
                # collect valid
                valid_bones = [bone_ids[k] for k in range(4) if bone_ids[k] >= 0 and raw_wts[k] > 0]
                valid_wts = [raw_wts[k] for k in range(4) if bone_ids[k] >= 0 and raw_wts[k] > 0]
                total = sum(valid_wts)
                if total > 0:
                    norm_wts = [ww / total for ww in valid_wts]
                    influences = list(zip(valid_bones, norm_wts))
                else:
                    influences = []
                    print(f"Warning: Vert {vert_idx} has no valid influences")
                mesh_poses[vert_idx] = pos
                mesh_uvs[vert_idx] = (u, v_uv)
                mesh_vert_influences[vert_idx] = influences
                max_vert_idx = max(max_vert_idx, vert_idx)
                i += 1
            i += 1  # End verts }
            vert_count = max_vert_idx + 1  # Assume 0-based contiguous

            # Parse tris
            mesh_tris = []
            while i < len(lines) and "tris " not in lines[i]: i += 1
            if i >= len(lines):
                print(f"Warning: No tris for mesh {m}")
                continue
            parts = lines[i].split()
            orig_tri_count = int(parts[1]) if len(parts) > 1 else 0
            i += 1  # Enter tris {
            max_tri_idx = -1
            tri_dict = {}
            while i < len(lines) and "}" not in lines[i]:
                line = lines[i]
                tri_match = re.search(r'tri\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
                if not tri_match:
                    i += 1
                    continue
                tri_idx, a, b, c = map(int, tri_match.groups())
                if a < vert_count and b < vert_count and c < vert_count:
                    tri_dict[tri_idx] = (a, b, c)
                else:
                    print(f"Warning: Invalid tri indices {a},{b},{c} for tri {tri_idx}")
                max_tri_idx = max(max_tri_idx, tri_idx)
                i += 1
            i += 1  # End tris }
            # Sort tris by index
            for t_idx in sorted(tri_dict.keys()):
                mesh_tris.append(tri_dict[t_idx])

            # Skip to end mesh }
            while i < len(lines) and "}" not in lines[i]: i += 1
            i += 1

            # Compute per-mesh weights with full inv bind
            mesh_weights = []
            vert_starts = [0] * vert_count
            current_w_idx = 0
            for v in range(vert_count):
                if v not in mesh_poses:
                    print(f"Warning: Missing vert {v}, skipping")
                    continue
                vert_starts[v] = current_w_idx
                for bone_id, wt in mesh_vert_influences.get(v, []):
                    if bone_id < len(joints):
                        bind_mat = bind_mats[bone_id]
                        inv_bind = mat4_inv(bind_mat)
                        offset = mat_mult_vec(inv_bind, mesh_poses[v])
                        mesh_weights.append({'joint': bone_id, 'bias': wt, 'offset': offset})
                        current_w_idx += 1
                    else:
                        print(f"Warning: Invalid bone {bone_id} for vert {v}")

            # Output mesh
            if vert_count > 0 and len(mesh_tris) > 0:
                out.write("mesh {\n")
                out.write(f'shader "{shader}"\n')
                out.write(f"  numverts {vert_count}\n")
                for v in range(vert_count):
                    if v not in mesh_uvs:
                        print(f"Warning: Missing UV for vert {v}, using default")
                        u, v_uv = 0.0, 0.0
                    else:
                        u, v_uv = mesh_uvs[v]
                    count = len(mesh_vert_influences.get(v, []))
                    start = vert_starts[v]
                    out.write(f"  vert {v} ( {u:.6f} {v_uv:.6f} ) {start} {count}\n")
                out.write(f"  numtris {len(mesh_tris)}\n")
                for t_idx, (a, b, c) in enumerate(mesh_tris):
                    out.write(f"  tri {t_idx} {a} {b} {c}\n")
                out.write(f"  numweights {len(mesh_weights)}\n")
                for w_idx, w in enumerate(mesh_weights):
                    j_id = w['joint']
                    bias = w['bias']
                    ox, oy, oz = w['offset']
                    out.write(f"  weight {w_idx} {j_id} {bias:.6f} ( {ox:.6f} {oy:.6f} {oz:.6f} )\n")
                out.write("}\n\n")
            else:
                print(f"Warning: Skipping empty mesh {m}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python md6_to_md5_mesh.py input.md6mesh output.md5mesh")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
    print("Mesh conversion complete!")
