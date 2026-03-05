import pygame
import numpy as np
from OpenGL import GL

from rendering.lighting_renderer import LightRenderer, RenderObject
from rendering.debug_renderer import DebugRenderer

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
from components.light_component import LightComponent

from debug.gizmo import DebugGizmo
from debug.object_control import DebugObjectController
from debug.ui.debug_window import UIManager


# ============================================================
# Pygame / OpenGL Initialization
# ============================================================

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


# ============================================================
# Core Engine Objects
# ============================================================

clock = pygame.time.Clock()
input_state = InputState()
physics = PhysicsWorld()
player = Player()
camera = Camera(player, physics)
renderer = LightRenderer(width=WIDTH, height=HEIGHT)
debug_renderer = DebugRenderer()
audio = AudioEngine()
world = World(audio, "engine/world_gen.json")

gizmo = DebugGizmo()
debug_controller = DebugObjectController()
debug_ui = UIManager()


# ============================================================
# Static Physics Plane
# ============================================================

plane_game_object = GameObject(
    mesh=None,
    transform=Transform(position=(0.0, 0.05, 0.0)),
    material=None,
    collider=AABBCollider(size=(1000.0, 0.1, 1000.0)),
)

physics.add_static(plane_game_object)

plane_mesh = Mesh(plane_vertices)


# ============================================================
# Register World Colliders
# ============================================================

for obj in world.objects:
    if obj.collider is not None:
        physics.add_static(obj)


# ============================================================
# Load Player Mannequin (FBX)
# ============================================================

mesh, skeleton, material, animations = create_mannequin_from_fbx("assets/models/Frank")

mannequin = Mannequin(
    player=player,
    mesh=mesh,
    material=Material(color=(1.0, 1.0, 1.0)),
    skeleton=skeleton,
    foot_offset=0.0,
    scale=0.018,
)


# ============================================================
# Animator Setup
# ============================================================

anim_clip = list(animations.values())[0]
mannequin.animator = Animator(skeleton=mannequin.skeleton, clip=anim_clip, loop=True)


# ============================================================
# Scene Object List (Render Layer)
# ============================================================

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


# ============================================================
# Main Loop
# ============================================================

running = True

while running:
    dt = clock.tick(240) / 1000.0

    # ------------------------------------------------------------
    # Event Handling
    # ------------------------------------------------------------

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                gizmo.toggle()

    mx, my = pygame.mouse.get_rel()
    player.process_mouse(mx, my)

    actions = input_state.update()

    # ------------------------------------------------------------
    # Player + Physics
    # ------------------------------------------------------------

    player.prev_position = player.position.copy()
    player.process_keyboard(actions, dt)
    physics.step(dt, player)

    for obj in world.objects:
        obj.update(dt)

    audio.update(camera)

    # ------------------------------------------------------------
    # Light Synchronization
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Render Pipeline
    # ------------------------------------------------------------

    renderer.render_shadow_pass(scene_objects, avatars=[])
    renderer.render_ssao_pass(camera, scene_objects)

    renderer.render_final_pass(
        None,
        player,
        camera,
        scene_objects,
        WIDTH,
        HEIGHT,
        debug_renderer=debug_renderer,
    )

    renderer.render_volumetric_pass(camera)
    renderer.render_bloom_pass()

    # ------------------------------------------------------------
    # Debug Object Control
    # ------------------------------------------------------------

    target_transform = debug_controller.update(world.objects)

    if target_transform is not None:
        object_position = target_transform.position
        object_scale = target_transform.scale
    else:
        object_position = (0.0, 0.0, 0.0)
        object_scale = (0.0, 0.0, 0.0)

    current_name = "None"
    if debug_controller.targets:
        current_obj = debug_controller.targets[debug_controller.current_index]
        current_name = getattr(current_obj, "obj_name", None)
        if not current_name:
            current_name = type(current_obj).__name__

    debug_renderer.render_debug_hud(
        clock,
        player,
        {"target": current_name},
        object_position,
        object_scale,
        extra_lines=debug_ui.get_lines(),
    )

    # ------------------------------------------------------------
    # Collider Gizmos
    # ------------------------------------------------------------

    if gizmo.enabled:
        vp = camera.get_projection_matrix(WIDTH / HEIGHT) @ camera.get_view_matrix()

        for obj in physics.static_objects:
            if hasattr(obj, "collider") and obj.collider is not None:
                corners = obj.collider.get_corners(obj.transform)

                edges = [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 0),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 4),
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                ]

                lines = []
                for i0, i1 in edges:
                    lines.append(corners[i0])
                    lines.append(corners[i1])

                selected_obj = None
                if debug_controller.targets:
                    selected_obj = debug_controller.targets[
                        debug_controller.current_index
                    ]

                if obj is selected_obj:
                    color = (1, 0, 0)
                else:
                    color = (0, 1, 0)

                gizmo.draw_lines(vp, np.array(lines), color=color)

    pygame.display.flip()


# ============================================================
# Shutdown
# ============================================================

audio.shutdown()
pygame.quit()
