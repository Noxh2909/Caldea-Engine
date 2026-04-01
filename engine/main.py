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
from gameobjects.mesh import Mesh
from gameobjects.collider.aabb import AABBCollider
from gameobjects.object import GameObject
from gameobjects.vertec import plane_vertices

from audio.audio_enigne import AudioEngine
from components.light_component import LightComponent

from debug.gizmo import DebugGizmo
from debug.object_control import DebugObjectController
from debug.ui.debug_window import UIManager

# ============================================================
# Initialization Function
# ============================================================


def initialize():
    # Pygame / OpenGL Initialization
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

    # Core Engine Objects
    clock = pygame.time.Clock()
    input_state = InputState()
    physics = PhysicsWorld()
    player = Player()
    camera = Camera(player, physics)
    renderer = LightRenderer(camera=camera, width=WIDTH, height=HEIGHT)
    debug_renderer = DebugRenderer()
    audio = AudioEngine()
    world = World(audio, "engine/world_gen.json")
    
    # ------------------------------------------------------------
    # Test objects: spawn 2000 cubes at height 15
    # ------------------------------------------------------------
    # cube_mesh = MeshRegistry.get("cube")

    # for i in range(2000):
    #     obj = GameObject(
    #         mesh=cube_mesh,
    #         transform=Transform(position=(i * 2.0, 15.0, 0.0)),
    #         material=Material(color=(0.8, 0.8, 0.8), shininess=16, specular_strength=0.3),
    #         collider=None,
    #         obj_name="test_cube",
    #     )
    #     world.objects.append(obj)
        
    gizmo = DebugGizmo()
    debug_controller = DebugObjectController()
    debug_ui = UIManager()
    ui_focus = False
    debug_enabled = False
    debug_ui.enabled = False
    gizmo.enabled = False
    debug_renderer.debug_enabled = False

    # ------------------------------------------------------------
    # Physics Setup
    # ------------------------------------------------------------

    # Static Physics Plane
    plane_game_object = GameObject(
        mesh=None,
        transform=Transform(position=(0.0, 0.05, 0.0)),
        material=None,
        collider=AABBCollider(size=(100.0, 1.0, 100.0)),
    )
    physics.add_static(plane_game_object)
    plane_mesh = Mesh(plane_vertices)

    # Register World Colliders
    for obj in world.objects:   
        if obj.collider is not None:
            physics.add_static(obj)
            
    for obj in world.objects:
        if getattr(obj, "use_gravity", False):
            physics.add_dynamic(obj)

    # Scene Object List (Render Layer)
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

    # Debug UI Object Name List
    world_object_names = []
    for obj in world.objects:
        name = getattr(obj, "obj_name", None)
        if not name:
            name = type(obj).__name__
        world_object_names.append(name)

    return {
        "clock": clock,
        "input_state": input_state,
        "physics": physics,
        "player": player,
        "camera": camera,
        "renderer": renderer,
        "debug_renderer": debug_renderer,
        "audio": audio,
        "world": world,
        "gizmo": gizmo,
        "debug_controller": debug_controller,
        "debug_ui": debug_ui,
        # "scene_objects": scene_objects,
        "plane_mesh": plane_mesh,
        "ui_focus": ui_focus,
        "debug_enabled": debug_enabled,
        "WIDTH": WIDTH,
        "HEIGHT": HEIGHT,
    }


# ============================================================
# Main Loop Function
# ============================================================


def main_loop(engine):
    running = True
    while running:
        dt = engine["clock"].tick(240) / 1000.0

        # ------------------------------------------------------------
        # Event Handling
        # ------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    engine["debug_enabled"] = not engine["debug_enabled"]

                    engine["debug_renderer"].debug_enabled = engine["debug_enabled"]
                    engine["debug_ui"].enabled = engine["debug_enabled"]
                    engine["gizmo"].enabled = engine["debug_enabled"]

                    if not engine["debug_enabled"]:
                        engine["ui_focus"] = False
                        pygame.mouse.set_visible(False)
                        pygame.event.set_grab(True)

                if event.key == pygame.K_ESCAPE and engine["debug_enabled"]:
                    engine["ui_focus"] = not engine["ui_focus"]
                    pygame.mouse.set_visible(engine["ui_focus"])
                    pygame.event.set_grab(not engine["ui_focus"])
            # Regain focus on mouse click if UI focus is enabled
            if event.type == pygame.MOUSEBUTTONDOWN:
                if engine["ui_focus"]:
                    engine["ui_focus"] = False
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True)
                    pygame.mouse.get_rel()

        if not engine["ui_focus"]:
            mx, my = pygame.mouse.get_rel()
            engine["player"].process_mouse(mx, my)
        else:
            pygame.mouse.get_rel()

        actions = engine["input_state"].update()

        # ------------------------------------------------------------
        # Player + Physics
        # ------------------------------------------------------------
        engine["player"].prev_position = engine["player"].position.copy()
        engine["player"].process_keyboard(actions, dt)
        engine["physics"].step(dt, engine["player"])

        for obj in engine["world"].objects:
            obj.update(dt)

        try:
            engine["audio"].update(engine["camera"])
        except Exception:
            pass

        # ------------------------------------------------------------
        # Light Synchronization
        # ------------------------------------------------------------
        for obj in engine["world"].objects:
            light_comp = obj.get_component(LightComponent)
            if light_comp:
                data = light_comp.get_light_data()
                if data:
                    engine["renderer"].light_pos = data["position"]
                    engine["renderer"].light_color = data["color"]
                    engine["renderer"].light_intensity = data["intensity"]
                    engine["renderer"].light_ambient = data["ambient"]

        light_space_matrix = engine["renderer"].point_light_matrices()

        # ------------------------------------------------------------
        # Frustum Culling
        # ------------------------------------------------------------
        aspect = engine["WIDTH"] / engine["HEIGHT"]
        planes = engine["camera"].get_frustum_planes(aspect)

        visible_objects = []
        for obj in engine["world"].objects:

            # -------------------------------------------------
            # 1. Collider based culling (preferred)
            # -------------------------------------------------
            if obj.collider is not None:
                min_corner, max_corner = obj.collider.get_bounds(obj.transform)

            # -------------------------------------------------
            # 2. Mesh based fallback if no collider
            # -------------------------------------------------
            elif obj.mesh is not None and hasattr(obj.mesh, "vertex_positions"):
                verts = obj.mesh.vertex_positions

                vmin = verts.min(axis=0)
                vmax = verts.max(axis=0)

                scale = np.array(obj.transform.scale)
                pos = np.array(obj.transform.position)

                min_corner = vmin * scale + pos
                max_corner = vmax * scale + pos

            # -------------------------------------------------
            # 3. No collider and no mesh → cannot cull
            # -------------------------------------------------
            else:
                visible_objects.append(obj)
                continue

            if obj.collider.aabb_in_frustum(planes, min_corner, max_corner):
                visible_objects.append(obj)

        # ------------------------------------------------------------
        # Render Pipeline
        # ------------------------------------------------------------
        engine["renderer"].render_shadow_pass(engine["world"].objects, avatars=[])
        engine["renderer"].render_ssao_pass(engine["camera"], visible_objects)

        engine["renderer"].render_final_pass(
            engine["player"],
            engine["camera"],
            visible_objects,
            engine["WIDTH"],
            engine["HEIGHT"],
            debug_renderer=engine["debug_renderer"],
        )

        engine["renderer"].render_volumetric_pass(engine["camera"])
        engine["renderer"].render_bloom_pass()

        # ------------------------------------------------------------
        # Debug Object Control
        # ------------------------------------------------------------
        target_transform = engine["debug_controller"].update(engine["world"].objects)

        if target_transform is not None:
            object_position = target_transform.position
            object_scale = target_transform.scale
        else:
            object_position = (0.0, 0.0, 0.0)
            object_scale = (0.0, 0.0, 0.0)

        current_name = "None"
        if engine["debug_controller"].targets:
            current_obj = engine["debug_controller"].targets[
                engine["debug_controller"].current_index
            ]
            current_name = getattr(current_obj, "obj_name", None)
            if not current_name:
                current_name = type(current_obj).__name__

        yaw = 0.0
        pitch = 0.0
        roll = 0.0

        if target_transform is not None:
            # Use rotation instead of position
            pitch = target_transform.pitch
            yaw = target_transform.yaw
            roll = target_transform.roll

        engine["debug_renderer"].render_debug_hud(
            engine["clock"],
            engine["player"],
            {
                "target": current_name,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
            },
            object_position,
            object_scale,
        )

        # ------------------------------------------------------------
        # Collider Gizmos
        # ------------------------------------------------------------
        if engine["gizmo"].enabled:
            vp = (
                engine["camera"].get_projection_matrix(
                    engine["WIDTH"] / engine["HEIGHT"]
                )
                @ engine["camera"].get_view_matrix()
            )

            for obj in engine["world"].objects:
                if hasattr(obj, "collider") and obj.collider is not None:
                    corners = obj.collider.get_corners(obj.transform)

                    # fmt: off
                    edges = [
                        (0, 1), (1, 2), (2, 3), (3, 0),
                        (4, 5), (5, 6), (6, 7), (7, 4),
                        (0, 4), (1, 5), (2, 6), (3, 7),
                    ]        
                    # fmt: on

                    lines = []
                    for i0, i1 in edges:
                        lines.append(corners[i0])
                        lines.append(corners[i1])

                    selected_obj = None
                    if engine["debug_controller"].targets:
                        selected_obj = engine["debug_controller"].targets[
                            engine["debug_controller"].current_index
                        ]

                    if obj is selected_obj:
                        color = (1, 0, 0)
                    else:
                        color = (0, 1, 0)

                    engine["gizmo"].draw_lines(vp, np.array(lines), color=color)

        if engine["debug_enabled"]:
            mouse_pos = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()[0]

            engine["debug_ui"].update(mouse_pos, mouse_pressed)

            # draw on pygame surface overlay
            screen_surface = pygame.display.get_surface()
            engine["debug_ui"].draw(screen_surface)

        pygame.display.flip()

    # ============================================================
    # Shutdown
    # ============================================================
    engine["audio"].shutdown()
    pygame.quit()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    engine = initialize()
    main_loop(engine)
