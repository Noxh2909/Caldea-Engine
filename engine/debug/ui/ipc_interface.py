from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
)
from PySide6.QtCore import Qt
import json
import sys
import os
import socket

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
        layout.setContentsMargins(4, 0, 4, 0)
        # layout.setSpacing(3)
        self.setLayout(layout)

        # Show only last part of path as label
        clean_label = label.split('.')[-1]
        clean_label = clean_label.split('[')[0]

        self.label = QLabel(clean_label)
        self.label.setFixedWidth(160)

        self.field = QLineEdit(self._format_value(value))
        self.field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.field.setFixedSize(150, 24)

        self.field.editingFinished.connect(self._commit)

        layout.addWidget(self.label)
        layout.addWidget(self.field)

        self._value = value

    def _format_value(self, value):
        if isinstance(value, float):
            return f"{value:.1f}"
        if isinstance(value, list):
            return str([round(v,1) if isinstance(v,float) else v for v in value])
        return str(value)

    def _commit(self):
        text = self.field.text()
        try:
            if text.startswith('['):
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

        # Load world JSON
        if os.path.exists(WORLD_PATH):
            with open(WORLD_PATH, "r") as f:
                self.loaded_data[WORLD_PATH] = json.load(f)
        else:
            self.loaded_data[WORLD_PATH] = {"objects": []}

        # Dropdown for objects
        self.object_dropdown = QComboBox()
        self.object_dropdown.setFixedHeight(30)
        main_layout.addWidget(self.object_dropdown)

        self.object_dropdown.currentIndexChanged.connect(self._reload_object_controls)

        # Container for value controls
        self.controls_container = QVBoxLayout()
        self.controls_container.setSpacing(2)
        self.controls_container.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(self.controls_container)

        # Save button
        self.save_button = QPushButton("Save JSON To Disk")
        self.save_button.setFixedHeight(30)
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
        # OBJECT SECTION
        # ----------------------------
        self._add_section_header("Objekt:")

        for key in ["obj_name", "mesh"]:
            if key in obj:
                self._create_control(f"objects[{index}].{key}", obj[key])

        # ----------------------------
        # MATERIAL SECTION
        # ----------------------------
        if "material" in obj:
            self._add_section_header("Material:")
            material = obj["material"]
            for key, value in material.items():
                self._create_control(
                    f"objects[{index}].material.{key}", value
                )

        # ----------------------------
        # PROPERTIES SECTION
        # ----------------------------
        self._add_section_header("Properties:")

        for key in obj:
            if key not in ["obj_name", "mesh", "material"]:
                self._create_control(
                    f"objects[{index}].{key}", obj[key]
                )

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
        with open(WORLD_PATH, "w") as f:
            json.dump(self.loaded_data[WORLD_PATH], f, indent=2)


# ------------------------------------------------------------
# STANDALONE
# ------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DebugInterface()
    window.show()
    sys.exit(app.exec())
