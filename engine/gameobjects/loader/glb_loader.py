from PIL import Image
import io
from pygltflib import GLTF2
import numpy as np
import struct
from typing import Optional

class GLBLoader:
    def __init__(self, path: str):
        gltf = GLTF2().load(path)
        if gltf is None:
            raise ValueError(f"Failed to load GLTF: {path}")
        self.gltf: GLTF2 = gltf

        binary = self.gltf.binary_blob()
        if binary is None:
            raise ValueError("GLTF has no binary blob (use .glb)")
        self._binary: bytes = binary

    # -------------------------------------------------
    # Accessor helpers
    # -------------------------------------------------
    def _component_count(self, accessor_type: str) -> int:
        return {
            "SCALAR": 1,
            "VEC2": 2,
            "VEC3": 3,
            "VEC4": 4,
            "MAT4": 16,
        }[accessor_type]

    def _component_format(self, component_type: int):
        if component_type == 5126:  # FLOAT
            return "f", 4
        if component_type == 5125:  # UNSIGNED_INT
            return "I", 4
        if component_type == 5123:  # UNSIGNED_SHORT
            return "H", 2
        if component_type == 5121:  # UNSIGNED_BYTE
            return "B", 1
        raise ValueError(f"Unsupported component type: {component_type}")

    def _read_accessor(self, acc_index: int) -> np.ndarray:
        acc = self.gltf.accessors[acc_index]
        if acc.bufferView is None:
            raise ValueError("Accessor has no bufferView")

        view = self.gltf.bufferViews[acc.bufferView]
        comp_char, comp_size = self._component_format(acc.componentType)
        comps = self._component_count(acc.type)

        stride = view.byteStride or (comp_size * comps)
        offset = (view.byteOffset or 0) + (acc.byteOffset or 0)

        fmt = "<" + comp_char * comps
        out = np.empty((acc.count, comps), dtype=np.float32)

        buf: bytes = self._binary  # type narrowing for static checkers
        for i in range(acc.count):
            out[i] = struct.unpack_from(fmt, buf, offset + i * stride)

        return out

    # -------------------------------------------------
    # Animation helpers
    # -------------------------------------------------
    def _build_local_matrix(self, translation, rotation, scale):
        """Build a 4x4 transformation matrix from TRS components."""
        # Create scale matrix
        S = np.diag([scale[0], scale[1], scale[2], 1.0]).astype(np.float32)
        
        # Create rotation matrix from quaternion (x, y, z, w)
        x, y, z, w = rotation
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y), 0],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x), 0],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Create translation matrix
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = translation
        
        # Combine: T * R * S
        return T @ R @ S

    def _read_animation_sampler(self, sampler):
        if sampler.input is None or sampler.output is None:
            return None, None

        times = self._read_accessor(sampler.input).flatten()
        values = self._read_accessor(sampler.output)
        return times.astype(np.float32), values.astype(np.float32)

    # -------------------------------------------------
    # Texture helpers
    # -------------------------------------------------
    def _load_basecolor_texture(self, prim) -> Optional[Image.Image]:
        if prim.material is None:
            return None

        material = self.gltf.materials[prim.material]
        if material.pbrMetallicRoughness is None:
            return None

        tex_info = material.pbrMetallicRoughness.baseColorTexture
        if tex_info is None:
            return None

        texture = self.gltf.textures[tex_info.index]
        image = self.gltf.images[texture.source]

        if image.bufferView is None:
            raise ValueError("External textures (.gltf) not supported yet")

        view = self.gltf.bufferViews[image.bufferView]
        offset = view.byteOffset or 0
        length = view.byteLength

        img_bytes = self._binary[offset : offset + length]
        return Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def load_first_mesh(self) -> dict:
        if not self.gltf.meshes:
            raise ValueError("No meshes found in GLTF")

        mesh = self.gltf.meshes[0]
        prim = mesh.primitives[0]

        if prim.attributes.POSITION is None:
            raise ValueError("GLTF primitive has no POSITION attribute")

        positions = self._read_accessor(prim.attributes.POSITION)

        # --- Compute mesh bounds in model space ---
        min_bounds = positions.min(axis=0)
        max_bounds = positions.max(axis=0)

        model_height = max_bounds[1] - min_bounds[1]
        foot_offset = -min_bounds[1]  # distance from pivot to feet

        normals = (
            self._read_accessor(prim.attributes.NORMAL)
            if prim.attributes.NORMAL is not None
            else np.zeros_like(positions)
        )

        uvs = (
            self._read_accessor(prim.attributes.TEXCOORD_0)
            if prim.attributes.TEXCOORD_0 is not None
            else np.zeros((positions.shape[0], 2), dtype=np.float32)
        )

        indices = None
        if prim.indices is not None:
            idx = self._read_accessor(prim.indices)
            indices = idx.astype(np.uint32).flatten()

        vertices = np.hstack([positions, normals, uvs]).astype(np.float32)
        albedo = self._load_basecolor_texture(prim)

        # --- Debug: print animation summary ---
        if not self.gltf.animations:
            print("[GLB] No animations found.")
        else:
            print(f"[GLB] Found {len(self.gltf.animations)} animation(s):")
            for i, a in enumerate(self.gltf.animations):
                name = a.name if a.name else f"<unnamed_{i}>"
                ch = len(a.channels) if a.channels else 0
                smp = len(a.samplers) if a.samplers else 0
                print(f"  - {name}: channels={ch}, samplers={smp}")

        # --- Debug: detailed animation inspection (first animation only) ---
        if self.gltf.animations:
            anim0 = self.gltf.animations[0]
            print(f"[GLB] Inspecting animation: {anim0.name}")

            # Print first few channels with node names
            for i, ch in enumerate(anim0.channels[:10]):
                node_idx = ch.target.node
                node_name = (
                    self.gltf.nodes[node_idx].name
                    if node_idx is not None and self.gltf.nodes and self.gltf.nodes[node_idx].name
                    else f"<node_{node_idx}>"
                )
                print(
                    f"    Channel {i}: "
                    f"node={node_name} ({node_idx}), "
                    f"path={ch.target.path}, "
                    f"sampler={ch.sampler}"
                )

            # Compute animation duration from sampler inputs
            durations = []
            for sampler in anim0.samplers:
                if sampler.input is not None:
                    times = self._read_accessor(sampler.input).flatten()
                    if times.size > 0:
                        durations.append(times.max())

            if durations:
                print(f"[GLB] Animation duration: {max(durations):.3f} seconds")
            else:
                print("[GLB] Animation duration: unknown")

            # --- Debug: evaluate animation at fixed time t ---
            if self.gltf.animations:
                anim0 = self.gltf.animations[0]
                t = 0.5  # seconds
                print(f"[GLB] Evaluating animation at t={t:.2f}s")

                # --- Debug: Interpolated TRS at t=... ---
                trs_per_node = {}

                for i, ch in enumerate(anim0.channels[:10]):  # limit output
                    sampler = anim0.samplers[ch.sampler]

                    if sampler.input is None or sampler.output is None:
                        continue

                    times = self._read_accessor(sampler.input).flatten()
                    values = self._read_accessor(sampler.output)

                    if times.size == 0:
                        continue

                    # find last keyframe index with time <= t
                    idx = np.searchsorted(times, t, side="right") - 1
                    idx = max(0, min(idx, len(times) - 1))

                    node_idx = ch.target.node
                    node_name = (
                        self.gltf.nodes[node_idx].name
                        if node_idx is not None and self.gltf.nodes and self.gltf.nodes[node_idx].name
                        else f"<node_{node_idx}>"
                    )

                    v = values[idx]

                    print(
                        f"    node={node_name} ({node_idx}), "
                        f"path={ch.target.path}, "
                        f"key={idx}, "
                        f"value={v}"
                    )

                    node = ch.target.node
                    trs_per_node.setdefault(
                        node,
                        {
                            "translation": np.zeros(3, dtype=np.float32),
                            "rotation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                            "scale": np.ones(3, dtype=np.float32),
                        },
                    )

                    trs_per_node[node][ch.target.path] = v

                # --- Debug: build local bone matrix for Hips ---
                for node_idx, trs in trs_per_node.items():
                    node_name = (
                        self.gltf.nodes[node_idx].name
                        if self.gltf.nodes[node_idx].name
                        else f"<node_{node_idx}>"
                    )

                    if node_name == "mixamorig1:Hips":
                        M_local = self._build_local_matrix(
                            translation=trs["translation"],
                            rotation=trs["rotation"],
                            scale=trs["scale"],
                        )

                        print("[GLB] Local bone matrix for Hips:")
                        print(M_local)

                # --- Build parent map (once) ---
                parent_map = {}
                for parent_idx, node in enumerate(self.gltf.nodes):
                    if node.children:
                        for child in node.children:
                            parent_map[child] = parent_idx

                # --- Build local matrices for all collected nodes ---
                local_matrices = {}
                for n_idx, trs in trs_per_node.items():
                    local_matrices[n_idx] = self._build_local_matrix(
                        translation=trs["translation"],
                        rotation=trs["rotation"],
                        scale=trs["scale"],
                    )

                # --- Recursive global matrix computation ---
                def compute_global(n_idx):
                    # If this node has no local animation, treat it as identity
                    local = local_matrices.get(n_idx, np.eye(4, dtype=np.float32))

                    parent = parent_map.get(n_idx)
                    if parent is None:
                        return local

                    return compute_global(parent) @ local

                # --- Debug: print global matrices for a small chain ---
                print("[GLB] Global bone matrices (debug):")
                for name in ("mixamorig1:Hips", "mixamorig1:Spine", "mixamorig1:Spine1"):
                    for idx, node in enumerate(self.gltf.nodes):
                        if node.name == name and idx in local_matrices:
                            M_global = compute_global(idx)
                            print(f"  Global matrix for {name}:")
                            print(M_global)

                # --- Inverse Bind Matrices (Skinning prep) ---
                if self.gltf.skins:
                    skin = self.gltf.skins[0]

                    if skin.inverseBindMatrices is None:
                        print("[GLB] Skin has no inverseBindMatrices")
                    else:
                        inv_bind = self._read_accessor(skin.inverseBindMatrices)
                        joints = skin.joints or []

                        print("[GLB] Skinning matrices (debug):")

                        for i, joint_idx in enumerate(joints):
                            if joint_idx not in local_matrices:
                                continue

                            M_global = compute_global(joint_idx)
                            M_inv = inv_bind[i].reshape((4, 4))

                            M_skin = M_global @ M_inv

                            joint_name = (
                                self.gltf.nodes[joint_idx].name
                                if self.gltf.nodes[joint_idx].name
                                else f"<node_{joint_idx}>"
                            )

                            print(f"  Skin matrix for {joint_name}:")
                            print(M_skin)

                            # limit debug output
                            if i >= 2:
                                break

        # animations
        animations = []
        if self.gltf.animations:
            for anim in self.gltf.animations:
                anim_data = {
                    "name": anim.name,
                    "samplers": [],
                    "channels": [],
                }

                for sampler in anim.samplers:
                    times, values = self._read_animation_sampler(sampler)
                    anim_data["samplers"].append(
                        {
                            "times": times,
                            "values": values,
                            "interpolation": sampler.interpolation or "LINEAR",
                        }
                    )

                for ch in anim.channels:
                    anim_data["channels"].append(
                        {
                            "sampler": ch.sampler,
                            "node": ch.target.node,
                            "path": ch.target.path,
                        }
                    )

                animations.append(anim_data)

        return {
            "vertices": vertices,
            "indices": indices,
            "albedo": albedo,
            "animations": animations,
            "nodes": self.gltf.nodes or [],
            "skins": self.gltf.skins or [],
            "bounds_min": min_bounds.astype(np.float32),
            "bounds_max": max_bounds.astype(np.float32),
            "model_height": float(model_height),
            "foot_offset": float(foot_offset),
        }
