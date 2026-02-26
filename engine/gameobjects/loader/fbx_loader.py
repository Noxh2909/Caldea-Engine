import json
import numpy as np
from pathlib import Path

from gameobjects.mesh import Mesh
from gameobjects.material_lookup import Material
from gameobjects.texture import Texture
from gameobjects.player.animation import AnimationClip, Keyframe, AnimationTrack
from gameobjects.player.mannequin import Bone, Skeleton

# ------------------------------
# FBX Loader
# ------------------------------

class FBXAsset:
    def __init__(self, mesh, bones, textures, animations):
        self.mesh = mesh
        self.bones = bones              # raw bone data
        self.textures = textures
        self.animations = animations


def load_fbx_asset(base_path: str) -> FBXAsset:
    base = Path(base_path)

    # Expect directory with fixed filenames: mesh.json, skel.json, tex.json
    if not base.is_dir():
        raise RuntimeError(
            "FBX JSON loader expects a directory containing "
            "mesh.json / skel.json / tex.json"
        )

    mesh_path = base / "mesh.json"
    skel_path = base / "skel.json"
    tex_path  = base / "tex.json"

    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    if not skel_path.exists():
        raise FileNotFoundError(skel_path)

    # ------------------------------
    # Load mesh
    # ------------------------------
    with open(mesh_path, "r") as f:
        mesh_json = json.load(f)

    positions = np.asarray(mesh_json["positions"], dtype=np.float32)
    normals   = np.asarray(mesh_json["normals"], dtype=np.float32)
    uvs       = np.asarray(mesh_json["uvs"], dtype=np.float32)
    bone_ids  = np.asarray(mesh_json["bone_ids"], dtype=np.uint16)
    weights   = np.asarray(mesh_json["bone_weights"], dtype=np.float32)

    mesh = Mesh(
        positions=positions,
        normals=normals,
        uvs=uvs,
        bone_ids=bone_ids,
        bone_weights=weights,
    )

    # ------------------------------
    # Load skeleton as raw bones
    # ------------------------------
    with open(skel_path, "r") as f:
        skel_json = json.load(f)

    bones = []
    for b in skel_json["bones"]:
        bones.append({
            "name": b["name"],
            "parent": b["parent"],
            "inverse_bind": np.array(
                b["inverse_bind"], dtype=np.float32
            ).reshape(4, 4),
        })

    # ------------------------------
    # Load textures
    # ------------------------------
    textures = {}
    if tex_path.exists():
        with open(tex_path, "r") as f:
            textures = json.load(f)

    # ------------------------------
    # Load animations (*.anim.json)
    # ------------------------------
    animations = {}

    for anim_path in base.glob("anim.json"):
        with open(anim_path, "r") as f:
            anim_json = json.load(f)

        tracks = {}
        for bone_name, frames in anim_json.get("tracks", {}).items():
            keyframes = []
            for k in frames:
                keyframes.append(
                    Keyframe(
                        time=k["time"],
                        translation=np.array(k.get("t", [0, 0, 0]), dtype=np.float32),
                        rotation=np.array(k.get("r", [0, 0, 0, 1]), dtype=np.float32),
                    )
                )
            tracks[bone_name] = keyframes

        clip = AnimationClip(
            name=anim_json.get("name", anim_path.stem),
            duration=anim_json["duration"],
            fps=anim_json.get("fps", 30),
            tracks=tracks,
        )

        animations[clip.name] = clip

    return FBXAsset(
        mesh=mesh,
        bones=bones,
        textures=textures,
        animations=animations,
    )


def create_mannequin_from_fbx(base_path: str):

    asset = load_fbx_asset(base_path)

    bones = []
    for b in asset.bones:
        bones.append(
            Bone(
                name=b["name"],
                parent=b["parent"],
                inverse_bind=b["inverse_bind"],
            )
        )

    skeleton = Skeleton(bones)

    material = Material(color=(1.0, 1.0, 1.0))
    albedo_path = "assets/skins/mannequin_albedo.png"

    if albedo_path:
        material.texture = Texture.load_texture(albedo_path)

    return asset.mesh, skeleton, material, asset.animations