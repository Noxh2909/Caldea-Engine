from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QComboBox, QScrollArea
from PySide6.QtGui import QPixmap, QCursor
from PySide6.QtWidgets import QToolTip
from PySide6.QtCore import Qt
import json
import sys
import os
import socket
import re

WORLD_PATH = "engine/world_gen.json"
RENDERER_PATH = "engine/rendering/renderer_config.json"


# ------------------------------------------------------------
# IPC CLIENT
# ------------------------------------------------------------


class IPCClient:
    def __init__(self, host="127.0.0.1", port=5050):
        self.host = host
        self.port = port
        self.socket = None
        self.connect()

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
        except Exception:
            self.socket = None

    def send(self, data: dict):
        if not self.socket:
            return
        try:
            message = json.dumps(data).encode("utf-8")
            self.socket.sendall(message)
        except Exception:
            pass


# ------------------------------------------------------------
# VALUE CONTROL
# ------------------------------------------------------------


class JsonValueControl(QWidget):
    def __init__(self, label, value, on_change):
        super().__init__()

        self.setFixedHeight(32)
        self.on_change = on_change

        layout = QHBoxLayout()
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(10)
        self.setLayout(layout)

        # Show only last part of path as label
        clean_label = label.split(".")[-1]
        clean_label = clean_label.split("[")[0]

        # Fields where stepping with < > does not make sense
        no_step_fields = {"texture_scale_mode", "obj_name", "name", "double_sided", "path", "loop", "mesh", "gravity", "glb_path"}

        self.label = QLabel(clean_label)
        self.label.setFixedWidth(160)
        layout.addWidget(self.label)
        layout.addStretch(1)

        self.minus_btn = QPushButton("<")
        self.minus_btn.setFixedSize(28, 28)
        self.minus_btn.setStyleSheet("""
QPushButton {
    border-radius: 8px;
    border: 1px solid #444;
    background-color: #2a2a2a;
    color: white;
}
QPushButton:hover {
    background-color: #3a3a3a;
}
QPushButton:pressed {
    background-color: #505050;
}
""")

        self.field = QLineEdit(self._format_value(value))
        self.field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.field.setFixedSize(140, 28)
        self.field.setStyleSheet("""
QLineEdit {
    border-radius: 10px;
    border: 1px solid #444;
    background-color: #1e1e1e;
    color: white;
    padding: 2px 6px;
}
QLineEdit:focus {
    border: 1px solid #6aa9ff;
}
""")

        self.plus_btn = QPushButton(">")
        self.plus_btn.setFixedSize(28, 28)
        self.plus_btn.setStyleSheet("""
QPushButton {
    border-radius: 8px;
    border: 1px solid #444;
    background-color: #2a2a2a;
    color: white;
}
QPushButton:hover {
    background-color: #3a3a3a;
}
QPushButton:pressed {
    background-color: #505050;
}
""")

        self.field.editingFinished.connect(self._commit)

        self.minus_btn.clicked.connect(self._decrement)
        self.plus_btn.clicked.connect(self._increment)

        button_gap = 6

        # Always keep buttons in layout so alignment stays identical
        layout.addWidget(self.minus_btn)
        layout.addSpacing(button_gap)

        layout.addWidget(self.field)
        layout.addSpacing(button_gap)

        layout.addWidget(self.plus_btn)
        layout.addSpacing(button_gap)

        # ------------------------------------------------------------
        # Info icon (only for mesh field)
        # ------------------------------------------------------------
        self.info_label = QLabel("ⓘ")
        self.info_label.setFixedWidth(18)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setCursor(Qt.CursorShape.PointingHandCursor)

        # ensure tooltip events always work
        self.info_label.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips, True)
        self.info_label.setMouseTracking(True)
        self.info_label.setToolTipDuration(5000)
        self.info_label.setStyleSheet("""
QLabel {
    color: #8aaaff;
    font-weight: bold;
}
""")

        # Tooltip texts per field
        tooltip_map = {
            "mesh": "Name of the mesh asset loaded from MeshRegistry.\nExample: cube, sphere, cloth.",
            "double_sided": "Render the material on both sides of the surface.\nUseful for cloth or thin geometry.",
            "opacity": "Transparency level of the material.\nRange: 0 (fully transparent) to 1 (fully opaque).",
            "position": "World position of the object in the format [x, y, z].\n x=left/right, y=up/down, z=forward/backward.",
            "yaw": "Rotation around the vertical axis in degrees. Changes the direction the object is facing.",
            "shininess": "How shiny the material is. Higher values create smaller, sharper highlights.\nRange: 0 (dull) to 100+ (very shiny).\npseudo-reflective",
            "specular_strength": "Intensity of the specular highlight. Higher values make the highlight brighter.\nRange: 0 (no highlight) to 1 (full brightness).",
            "obj_name": "Unique identifier for this object in the world JSON.",
            "name": "Material name used from the MaterialRegistry.",
            "path": "Audio file used by the audio component.\nengine/audio/audiosamples/",
            "loop": "If enabled, the audio will continuously loop.",
            "scale": "Uniform scale of the object.\n[x, y, z], x=left/right, y=up/down, z=forward/backward.\nDefault is [1, 1, 1]. [2.0, 2.0, 2.0] would be double size, [0.5, 0.5, 0.5] would be half size.",
            "texture_scale_mode": "How the material scales the texture:\n- default: uses the base material's UV scale\n- triplanar: uses triplanar projection for texture mapping.",
            "texture_scale_value": "Uniform scale applied to the material's texture coordinates.\nOnly used if texture_scale_mode is 'triplanar' or 'default' with a base material that has a texture.",
            "gravity": "If enabled, the object will be affected by gravity in the physics simulation.",
            "ambient_strength": "Intensity of the ambient light component. Higher values brighten the overall scene.\nRange: 0 (no ambient light) to 1+ (very bright).",
            "direction": "Direction vector of the light source.\n[x, y, z], where x=left/right, y=up/down, z=forward/backward.\nFor directional lights, this defines the direction the light is pointing.\n[0, 1, 0] would be a light shining straight down.",
            "color": "Color of the light in RGB format.\nValues are between 0 and 1. [1, 1, 1] is white light, [1, 0, 0] is red light, [0, 1, 0] is green light, and [0, 0, 1] is blue light.",
            "intensity": "Brightness of the light source. Higher values create a brighter light.\nRange: 0 (no light) to 10+ (very bright).",
            "max_distance": "Maximum distance of sound attenuation. Beyond this distance, the sound will no longer be audible.\nOnly used by audio components.",
            "rolloff": "Determines how quickly the sound attenuates with distance. Higher values cause the sound to fade out more rapidly as you move away from the source.\nOnly used by audio components.",
            "fade_ratio": "Controls the smoothness of the sound fade as it approaches the max_distance. A higher fade ratio creates a more gradual fade-out.\nOnly used by audio components.",
        }

        self.info_text = tooltip_map.get(clean_label)

        # Force tooltip display on hover (macOS fix)
        if self.info_text:
            self.info_label.enterEvent = lambda event: QToolTip.showText(QCursor.pos(), self.info_text or "")

        # Always reserve icon space so layout never shifts
        layout.addWidget(self.info_label)

        # Only show the icon for tooltip fields but keep the space for all rows
        if self.info_text:
            self.info_label.setText("ⓘ")
        else:
            # keep placeholder width but make it invisible
            self.info_label.setText("")

        # For fields where stepping makes no sense, visually hide buttons
        if clean_label in no_step_fields:
            self.minus_btn.setDisabled(True)
            self.plus_btn.setDisabled(True)

            self.minus_btn.setStyleSheet("background-color: transparent; border: none;")
            self.plus_btn.setStyleSheet("background-color: transparent; border: none;")

        self._value = value

    def _format_value(self, value):
        if isinstance(value, float):
            return f"{value:.1f}"
        if isinstance(value, list):
            return str([round(v, 1) if isinstance(v, float) else v for v in value])
        return str(value)

    def _commit(self):
        text = self.field.text()
        try:
            if text.startswith("["):
                new_value = json.loads(text)
            else:
                # try float
                try:
                    new_value = float(text)
                except:
                    new_value = text
            self.on_change(new_value)
        except Exception:
            pass

    def _increment(self):
        try:
            text = self.field.text()

            # Vector like [1,2,3]
            if text.startswith("["):
                arr = json.loads(text)
                new_arr = []
                for v in arr:
                    if isinstance(v, (int, float)):
                        new_arr.append(round(v + 0.5, 1))
                    else:
                        new_arr.append(v)

                self.field.setText(str(new_arr))
                self._commit()
                return

            value = float(text)
            value = round(value + 0.5, 1)
            self.field.setText(str(value))
            self._commit()

        except:
            pass

    def _decrement(self):
        try:
            text = self.field.text()

            if text.startswith("["):
                arr = json.loads(text)
                new_arr = []
                for v in arr:
                    if isinstance(v, (int, float)):
                        new_arr.append(round(v - 0.5, 1))
                    else:
                        new_arr.append(v)

                self.field.setText(str(new_arr))
                self._commit()
                return

            value = float(text)
            value = round(value - 0.5, 1)
            self.field.setText(str(value))
            self._commit()

        except:
            pass


# ------------------------------------------------------------
# DEBUG INTERFACE
# ------------------------------------------------------------


class DebugInterface(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Caldea Debug Panel")
        self.setFixedSize(500, 800)

        self.ipc = IPCClient()
        self.loaded_data = {}

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Global tooltip styling
        self.setStyleSheet(self.styleSheet() + """
QToolTip {
    background-color: #2a2a2a;
    color: white;
    border: 1px solid #555;
    padding: 6px;
    margin: 0px;
    border-radius: 6px;
}
""")

        # ------------------------------------------------------------
        # TOP ICON
        # ------------------------------------------------------------
        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_path = "caldea_engine_icon.png"
        if os.path.exists(icon_path):
            pix = QPixmap(icon_path)

            # Only downscale if the icon is larger than the target size
            target_size = 160
            if pix.height() > target_size or pix.width() > target_size:
                pix = pix.scaled(target_size, target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

            icon_label.setPixmap(pix)

        main_layout.addSpacing(2)
        main_layout.addWidget(icon_label)
        main_layout.addSpacing(2)

        # Load world JSON
        if os.path.exists(WORLD_PATH):
            with open(WORLD_PATH, "r") as f:
                self.loaded_data[WORLD_PATH] = json.load(f)
        else:
            self.loaded_data[WORLD_PATH] = {"objects": []}

        main_layout.addSpacing(10)

        # Scene title above dropdown
        scene_label = QLabel("Scene:")
        scene_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        scene_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 2px;")
        main_layout.addWidget(scene_label)

        # Dropdown for objects
        self.object_dropdown = QComboBox()
        self.object_dropdown.setFixedHeight(30)
        main_layout.addWidget(self.object_dropdown)

        self.object_dropdown.currentIndexChanged.connect(self._reload_object_controls)

        # Scrollable container for controls
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.scroll_widget = QWidget()
        self.controls_container = QVBoxLayout(self.scroll_widget)
        self.controls_container.setSpacing(6)
        self.controls_container.setContentsMargins(4, 4, 4, 4)
        self.controls_container.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll_area.setWidget(self.scroll_widget)

        # prevent horizontal squeezing and keep layout stable
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        main_layout.addWidget(self.scroll_area)

        # Save button
        self.save_button = QPushButton("Save Config")
        self.save_button.setFixedHeight(34)
        self.save_button.setStyleSheet("""
QPushButton {
    border-radius: 12px;
    border: 1px solid #444;
    background-color: #2a2a2a;
    color: white;
    padding: 4px 10px;
}
QPushButton:hover {
    background-color: #3a3a3a;
}
QPushButton:pressed {
    background-color: #505050;
}
""")
        self.save_button.clicked.connect(self._save_all)
        main_layout.addWidget(self.save_button)

        self._populate_dropdown()
        self._reload_object_controls()

    # ------------------------------------------------------------
    # SECTION HEADER HELPER
    # ------------------------------------------------------------
    def _add_section_header(self, title):
        header = QLabel(title)
        header.setStyleSheet("font-weight: bold; margin-top: 4px; margin-bottom: 2px;")
        header.setFixedHeight(20)
        self.controls_container.addWidget(header)
        self.controls_container.addSpacing(0)

    # ------------------------------------------------------------
    # DROPDOWN
    # ------------------------------------------------------------

    def _populate_dropdown(self):
        self.object_dropdown.clear()
        objects = self.loaded_data[WORLD_PATH].get("objects", [])

        for obj in objects:
            name = obj.get("obj_name", "Unnamed")
            self.object_dropdown.addItem(name)

    # ------------------------------------------------------------
    # LOAD CONTROLS FOR SELECTED OBJECT
    # ------------------------------------------------------------
    def _reload_object_controls(self):
        # Clear existing controls
        while self.controls_container.count():
            child = self.controls_container.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.deleteLater()

        objects = self.loaded_data[WORLD_PATH].get("objects", [])
        index = self.object_dropdown.currentIndex()

        if index < 0 or index >= len(objects):
            return

        obj = objects[index]

        # ----------------------------
        # OBJECT NAME
        # ----------------------------
        self._add_section_header("Object")

        if "obj_name" in obj:
            self._create_control(f"objects[{index}].obj_name", obj["obj_name"])

        # ----------------------------
        # DYNAMIC PROPERTY BUILD
        # ----------------------------
        for key, value in obj.items():

            if key == "obj_name":
                continue

            # Nested dict (material, light, cloth, audio...)
            if isinstance(value, dict):
                self._add_section_header(key.capitalize())

                for sub_key, sub_value in value.items():
                    path = f"objects[{index}].{key}.{sub_key}"
                    self._create_control(path, sub_value)

            else:
                path = f"objects[{index}].{key}"
                self._create_control(path, value)

        # keep controls packed at the top so spacing stays consistent
        self.controls_container.addStretch()

    def _build_object_controls(self, data, current_path):
        if isinstance(data, dict):
            for key, value in data.items():
                full_path = f"{current_path}.{key}"

                if isinstance(value, (int, float, str, list)):
                    self._create_control(full_path, value)
                else:
                    self._build_object_controls(value, full_path)

    # ------------------------------------------------------------
    # CREATE CONTROL
    # ------------------------------------------------------------

    def _create_control(self, full_path, value):
        def on_change(new_value):
            self._update_json_by_path(full_path, new_value)
            self.ipc.send({"path": full_path, "value": new_value})

        control = JsonValueControl(full_path, value, on_change)
        self.controls_container.addWidget(control)

    # ------------------------------------------------------------
    # UPDATE JSON
    # ------------------------------------------------------------

    def _update_json_by_path(self, path, new_value):
        obj = self.loaded_data[WORLD_PATH]
        parts = path.replace("]", "").replace("[", ".").split(".")
        target = obj

        for p in parts[:-1]:
            if p.isdigit():
                target = target[int(p)]
            else:
                target = target[p]

        last = parts[-1]
        if last.isdigit():
            target[int(last)] = new_value
        else:
            target[last] = new_value

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    def _save_all(self):
        data = self.loaded_data[WORLD_PATH]

        formatted = json.dumps(data, indent=2, ensure_ascii=False, separators=(", ", ": "))

        # collapse numeric arrays into single line
        def collapse_array(match):
            values = match.group(1)
            values = values.replace("\n", " ")
            values = re.sub(r"\s+", " ", values).strip()
            return f"[{values}]"

        formatted = re.sub(r"\[\s*\n\s*([0-9\.\-,\s]+?)\s*\n\s*\]", collapse_array, formatted, flags=re.MULTILINE)

        formatted += "\n"

        with open(WORLD_PATH, "w") as f:
            f.write(formatted)


# ------------------------------------------------------------
# STANDALONE
# ------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DebugInterface()
    window.show()
    sys.exit(app.exec())
