import pygame
import numpy as np
from OpenGL import GL

from rendering.renderer import Renderer, RenderObject
from world import World
from physics.world_physics import PhysicsWorld
from input import InputState

from gameobjects.player.player import Player
from gameobjects.player.camera import Camera
from gameobjects.transform import Transform
from gameobjects.material_lookup import Material
from gameobjects.mesh import Mesh
from gameobjects.loader.fbx_loader import create_mannequin_from_fbx
from gameobjects.collider.aabb import AABBCollider
from gameobjects.object import GameObject
from gameobjects.player.mannequin import Mannequin
from gameobjects.player.animator import Animator
from gameobjects.vertec import plane_vertices

from audio.audio_enigne import AudioEngine
from audio.audio_source import AudioSource

from debug.gizmo import DebugGizmo


# ====================
# Pygame / OpenGL init
# ====================

pygame.init()
pygame.display.set_caption("Game Engine")

pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
)

WIDTH, HEIGHT = 1400, 800
pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
GL.glViewport(0, 0, WIDTH, HEIGHT)

pygame.mouse.set_visible(False)
pygame.event.set_grab(True)
pygame.mouse.get_rel()

version = GL.glGetString(GL.GL_VERSION)
if version:
    print("OpenGL:", version.decode())


# ====================
# Core engine objects
# ====================

clock = pygame.time.Clock()
input_state = InputState()
physics = PhysicsWorld()
player = Player()
camera = Camera(player, physics)
renderer = Renderer(width=WIDTH, height=HEIGHT)
world = World("engine/world_gen.json")
audio = AudioEngine()
gizmo = DebugGizmo()

# ====================
# Static Plane
# ====================
 
# Create Physics Plane
plane_game_object = GameObject(
    mesh=None,
    transform=Transform(position=(0.0, 0.05, 0.0)),
    material=None,
    collider=AABBCollider(size=(1000.0, 0.1, 1000.0)),
)

physics.add_static(plane_game_object)

# Create Render Plane
plane_mesh = Mesh(plane_vertices)

# ====================
# Register world colliders
# ====================

for obj in world.objects:
    if obj.collider is not None:
        physics.add_static(obj)

# ====================
# Create Audio Sources (JSON-driven)
# ====================

AUDIO_BASE_PATH = "engine/audio/audiosamples/"

for obj in world.objects:
    config = obj.audio_config
    if not config:
        continue

    file_name = config.get("path")
    if not file_name:
        continue

    full_path = AUDIO_BASE_PATH + file_name

    audio_source = AudioSource(
        path=full_path,
        position=obj.transform.position
    )

    # Volume
    volume_percent = config.get("volume")
    base_gain = max(0.0, min(1.0, volume_percent / 100.0))
    if audio_source.source is not None:
        audio_source.source.set_gain(base_gain)

    # Distance
    if audio_source.source is not None:
        audio_source.source.set_max_distance(config.get("max_distance"))
        audio_source.source.set_rolloff_factor(config.get("rolloff"))

    # Loop
    loop_enabled = config.get("loop")

    audio.add_source(audio_source)
    audio.object_sources[obj] = audio_source
    audio_source.play(loop=loop_enabled)


# ====================
# Sun / Light
# ====================

sun = world.sun
if sun and sun.light:
    renderer.set_light(
        position=sun.transform.position,
        direction=sun.light["direction"],
        color=sun.light["color"],
        intensity=sun.light["intensity"],
        ambient=sun.light.get("ambient_strength"),
    )


# ====================
# Load mannequin (FBX)
# ====================

mesh, skeleton, material, animations = create_mannequin_from_fbx("assets/models/Frank")

mannequin = Mannequin(
    player=player,
    mesh=mesh,
    material=Material(color=(1.0, 1.0, 1.0)),
    skeleton=skeleton,
    foot_offset=0.0,
    scale=0.018,
)
assert mannequin.mesh is not None, "Mannequin mesh not loaded"
assert mannequin.skeleton is not None, "Mannequin skeleton not loaded"


# ====================
# Animator
# ====================

assert animations, "No animations loaded from FBX"

# Nimm die erste Animation (z. B. Walk / Take 001)
anim_clip = list(animations.values())[0]

mannequin.animator = Animator(skeleton=mannequin.skeleton, clip=anim_clip, loop=True)

# ====================
# Scene object list
# ====================

scene_objects: list[RenderObject] = []

for obj in world.objects:
    if obj.mesh is not None:
        scene_objects.append(
            RenderObject(
                mesh=obj.mesh,
                transform=obj.transform,
                material=obj.material,
            )
        )


# ====================
# Main Loop
# ====================

running = True
first_person = True
camera.third_person = False

# State for object control
control_state = {"target": "sun", "m_was_pressed": False}

while running:
    dt = clock.tick(120) / 1000.0

    # -------------
    # Events
    # -------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                gizmo.toggle()

    mx, my = pygame.mouse.get_rel()
    player.process_mouse(mx, my)

    actions = input_state.update()

    if actions["toggle_third_person"]:
        first_person = not first_person
        camera.third_person = not first_person
    

    # -------------
    # Player + Physics
    # -------------
    player.prev_position = player.position.copy()
    player.process_keyboard(actions, dt)
    physics.step(dt, player)

    # -------------
    # Audio Update
    # -------------
    # Sync audio source positions + apply soft distance fade
    for obj, source in audio.object_sources.items():
        source.set_position(obj.transform.position)

        config = obj.audio_config
        if not config:
            continue

        max_distance = config.get("max_distance", 10.0)
        base_gain = config.get("volume", 100.0) / 100.0
        fade_ratio = config.get("fade_ratio", 0.4)

        source.apply_distance_fade(
            camera.player.position,
            max_distance,
            base_gain,
            fade_ratio
        )

    audio.update(camera)
    
    # -------------
    # update mannequin
    # -------------
    # if mannequin.animator is not None:
    #     mannequin.animator.update(dt)w

    # -------------
    # Render passes
    # -------------
    light_space_matrix = renderer.point_light_matrices()

    # Shadow pass
    renderer.render_shadow_pass(scene_objects, avatars=[])

    # SSAO pass
    renderer.render_ssao_pass(camera, scene_objects)

    # Final lighting pass
    renderer.render_final_pass(None, player, camera, scene_objects)

    # Debug grid
    renderer.draw_debug_grid(camera, WIDTH / HEIGHT, size=50.0)

    #volumetric light pass
    renderer.render_volumetric_pass(camera)
    
    # Bloom pass
    renderer.render_bloom_pass()

    # -------------
    # DEBUG OBJECT CONTROL
    # -------------
    keys = pygame.key.get_pressed()

    obj_movement_speed = 0.1

    # Toggle control target with 'M' key (single press)
    if keys[pygame.K_m] and not control_state["m_was_pressed"]:
        # Get list of controllable objects
        controllable_objects = ["sun"] + [
            f"scene_{i}" for i in range(len(scene_objects) - 1)
        ]
        current_index = controllable_objects.index(control_state["target"])
        next_index = (current_index + 1) % len(controllable_objects)
        control_state["target"] = controllable_objects[next_index]
        control_state["m_was_pressed"] = True
    elif not keys[pygame.K_m]:
        control_state["m_was_pressed"] = False

    control_target = control_state["target"]

    # Determine which object to control
    target_transform = None
    if control_target == "mannequin":
        # Mannequin doesn't have a direct transform attribute, skip it
        target_transform = None
    elif control_target == "sun" and sun is not None:
        target_transform = sun.transform
    elif control_target.startswith("scene_"):
        scene_index = int(control_target.split("_")[1])
        if scene_index < len(scene_objects) - 1:  # Exclude mannequin
            target_transform = scene_objects[scene_index].transform

    # Get object position / scale (HUD-safe)
    if target_transform is not None:
        object_position = target_transform.position
        object_scale = target_transform.scale
    else:
        object_position = (0.0, 0.0, 0.0)
        object_scale = (0.0, 0.0, 0.0)

    # Apply movement to target object
    if target_transform is not None:
        if keys[pygame.K_UP]:
            target_transform.position[2] -= obj_movement_speed
        if keys[pygame.K_DOWN]:
            target_transform.position[2] += obj_movement_speed
        if keys[pygame.K_LEFT]:
            target_transform.position[0] -= obj_movement_speed
        if keys[pygame.K_RIGHT]:
            target_transform.position[0] += obj_movement_speed
        if keys[pygame.K_PAGEUP]:
            target_transform.position[1] += obj_movement_speed
        if keys[pygame.K_PAGEDOWN]:
            target_transform.position[1] -= obj_movement_speed

    # Debug HUD
    renderer.render_debug_hud(
        clock,
        player,
        obj=control_state,
        obj_pos=object_position,
        obj_scale=object_scale,
    )
    
    # -------------
    # Collider Gizmos
    # -------------
    if gizmo.enabled:
        vp = camera.get_projection_matrix(WIDTH / HEIGHT) @ camera.get_view_matrix()

        for obj in physics.static_objects:
            if hasattr(obj, "collider") and obj.collider is not None:
                corners = obj.collider.get_corners(obj.transform)

                edges = [
                    (0,1),(1,2),(2,3),(3,0),
                    (4,5),(5,6),(6,7),(7,4),
                    (0,4),(1,5),(2,6),(3,7)
                ]

                lines = []
                for i0, i1 in edges:
                    lines.append(corners[i0])
                    lines.append(corners[i1])

                gizmo.draw_lines(vp, np.array(lines), color=(0, 1, 0))

    pygame.display.flip()

audio.shutdown()
pygame.quit()
