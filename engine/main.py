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

from debug.gizmo import DebugGizmo
from debug.object_control import DebugObjectController
from components.light_component import LightComponent


# ====================
# Pygame / OpenGL init
# ====================

pygame.init()
pygame.display.set_caption("Caldea Engine")

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
audio = AudioEngine()
world = World(audio, "engine/world_gen.json")
gizmo = DebugGizmo()
debug_controller = DebugObjectController()

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

# ====================
# Animator
# ====================

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

while running:
    dt = clock.tick(240) / 1000.0

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

    # -------------
    # Player + Physics
    # -------------
    player.prev_position = player.position.copy()
    player.process_keyboard(actions, dt)
    physics.step(dt, player)    

    # Update world components
    for obj in world.objects:
        obj.update(dt)

    # Update audio listener (camera position & orientation)
    audio.update(camera)

    # -------------
    # -------------
    # if mannequin.animator is not None:
    #     mannequin.animator.update(dt)

    # -------------
    # Render passes
    # -------------
    # Collect LightComponents and update renderer light data

    for obj in world.objects:
        light_comp = obj.get_component(LightComponent)
        if light_comp:
            data = light_comp.get_light_data()
            if data:
                renderer.light_pos = data["position"]
                renderer.light_color = data["color"]
                renderer.light_intensity = data["intensity"]
                renderer.light_ambient = data["ambient"]

    light_space_matrix = renderer.point_light_matrices()

    # Shadow pass
    renderer.render_shadow_pass(scene_objects, avatars=[])

    # SSAO pass
    renderer.render_ssao_pass(camera, scene_objects)

    # Final lighting pass
    renderer.render_final_pass(None, player, camera, scene_objects, WIDTH, HEIGHT)

    # #volumetric light pass
    renderer.render_volumetric_pass(camera)
    
    # # Bloom pass
    renderer.render_bloom_pass()

    # -------------
    # DEBUG OBJECT CONTROL (generic)
    # -------------
    target_transform = debug_controller.update(world.objects)

    if target_transform is not None:
        object_position = target_transform.position
        object_scale = target_transform.scale
    else:
        object_position = (0.0, 0.0, 0.0)
        object_scale = (0.0, 0.0, 0.0)

    # Resolve current object name for HUD
    current_name = "None"
    if debug_controller.targets:
        current_obj = debug_controller.targets[debug_controller.current_index]
        current_name = getattr(current_obj, "obj_name", None)
        if not current_name:
            current_name = type(current_obj).__name__

    renderer.render_debug_hud(
        clock,
        player,
        obj={"target": current_name},
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

                # Get currently selected object
                selected_obj = None
                if debug_controller.targets:
                    selected_obj = debug_controller.targets[debug_controller.current_index]

                # Choose color
                if obj is selected_obj:
                    color = (1, 0, 0)
                else:
                    color = (0, 1, 0)

                gizmo.draw_lines(vp, np.array(lines), color=color)

    pygame.display.flip()

audio.shutdown()
pygame.quit()
