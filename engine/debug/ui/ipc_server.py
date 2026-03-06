import json
from PySide6.QtWidgets import QWidget, QVBoxLayout
from debug.ui.ipc_interface import JsonValueControl


class IPCInterface(QWidget):
    def __init__(self, ipc, path):
        super().__init__()
        self.ipc = ipc
        self.path = path
        self.container_layout = QVBoxLayout()
        self.setLayout(self.container_layout)
        self._load_section()

    def _load_section(self):
        with open(self.path, "r") as f:
            data = json.load(f)
        self._build_controls(data, self.path, data, current_path="")

    def _build_controls(self, data, file_path, root, current_path=""):
        if isinstance(data, dict):
            for key, value in data.items():
                full_path = f"{current_path}.{key}" if current_path else key

                if isinstance(value, (int, float)):
                    self._create_numeric_control(
                        full_path,
                        value,
                        file_path,
                        root,
                    )
                else:
                    self._build_controls(value, file_path, root, full_path)

        elif isinstance(data, list):
            for index, item in enumerate(data):
                full_path = f"{current_path}[{index}]"
                self._build_controls(item, file_path, root, full_path)

    def _create_numeric_control(self, full_path, value, file_path, root):
        def on_change(new_value):
            self._update_json_by_path(root, full_path, new_value)

            self.ipc.send({
                "path": full_path,
                "value": new_value
            })

        control = JsonValueControl(full_path, float(value), on_change)
        self.container_layout.addWidget(control)

    def _update_json_by_path(self, obj, path, new_value):
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
