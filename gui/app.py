import re
import sys, os
import json

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import QProcess, QProcessEnvironment
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, \
    QDoubleSpinBox, QPushButton, QPlainTextEdit, QTabWidget, QSizePolicy
import pyqtgraph as pg
from scipy.interpolate import make_interp_spline

SPARSITY_X_FIELD = "sparsity"
PARAMS_COUNT_X_FIELD = "params_count"

METRIC_REG_EXPS = {
    "original_model_parameters": re.compile(r'(?<=Original model parameters:\s)([\d,]+)'),
    "sliced_model_parameters": re.compile(r'(?<=Sliced model parameters:\s)([\d,]+)'),
    "perplexity": re.compile(r"(?<=Loaded model perplexity:\s)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "arc_challenge": re.compile(r'(?<="arc_challenge":\s)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'),
    "arc_easy": re.compile(r'(?<="arc_easy":\s)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'),
    "hellaswag": re.compile(r'(?<="hellaswag":\s)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'),
    "piqa": re.compile(r'(?<="piqa":\s)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'),
    "winogrande": re.compile(r'(?<="winogrande":\s)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'),
}


def get_models_graph_data() -> dict[str, dict]:
    file_dir_name = os.path.dirname(os.path.abspath(__file__))
    graph_data_dir_name = os.path.join(file_dir_name, "graph_data")
    models_dict = {}
    for root, dirs, files in os.walk(graph_data_dir_name):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        json_data = json.load(f)
                        json_data['file_path'] = file_path
                        org = os.path.split(root)[-1]
                        model_name = org + '/' + os.path.splitext(file)[0]
                        models_dict[model_name] = json_data
                except json.JSONDecodeError:
                    pass

    return models_dict


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_selection = None
        self.pruning_sparsity = .0
        self.process: QProcess | None = None
        self.process_output = QPlainTextEdit()
        self.process_output.setReadOnly(True)
        self.ansi_re = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        self.metric_res = METRIC_REG_EXPS
        self.graph_model_selection = None
        self.setWindowTitle("SliceGPT GUI")

        layout = QHBoxLayout()

        settings_and_output_layout = QVBoxLayout()
        settings_and_output_layout.addWidget(QLabel("Choose model:"))

        self.pruning_model_combo_box = QComboBox()
        self.models_dict = get_models_graph_data()
        models = sorted(self.models_dict.keys())

        if models:
            self.model_selection = models[0]

        self.pruning_model_combo_box.addItems(models)
        self.pruning_model_combo_box.currentTextChanged.connect(self.model_selection_changed)
        settings_and_output_layout.addWidget(self.pruning_model_combo_box)

        tabs = QTabWidget()
        tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        tabs.setMovable(True)

        pruning_layout = QVBoxLayout()
        sparsity_double_spin_box = QDoubleSpinBox()
        sparsity_double_spin_box.setRange(0, 0.99)
        sparsity_double_spin_box.setSingleStep(0.01)
        sparsity_double_spin_box.valueChanged.connect(self.pruning_sparsity_changed)
        settings_and_output_layout.addWidget(QLabel("Sparsity:"))
        settings_and_output_layout.addWidget(sparsity_double_spin_box)

        pruning_start_button = QPushButton("Start pruning")
        pruning_start_button.pressed.connect(self.start_pruning)
        pruning_layout.addWidget(pruning_start_button)

        pruning_widget = QWidget()
        pruning_widget.setLayout(pruning_layout)
        tabs.addTab(pruning_widget, "Pruning")

        perplexity_layout = QVBoxLayout()

        perplexity_eval_start_button = QPushButton("Evaluate perplexity")
        perplexity_eval_start_button.pressed.connect(self.evaluate_perplexity)
        perplexity_layout.addWidget(perplexity_eval_start_button)

        perplexity_eval_widget = QWidget()
        perplexity_eval_widget.setLayout(perplexity_layout)
        tabs.addTab(perplexity_eval_widget, "Perplexity")

        lm_evaluation_layout = QVBoxLayout()
        lm_evaluation_start_button = QPushButton("Evaluate language model")
        lm_evaluation_start_button.pressed.connect(self.evaluate_lm)
        lm_evaluation_layout.addWidget(lm_evaluation_start_button)

        lm_evaluation_widget = QWidget()
        lm_evaluation_widget.setLayout(lm_evaluation_layout)
        tabs.addTab(lm_evaluation_widget, "LM Evaluation")

        size_policy = QSizePolicy()
        size_policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        size_policy.setVerticalPolicy(QSizePolicy.Policy.Preferred)
        tabs.setSizePolicy(size_policy)

        settings_and_output_layout.addWidget(tabs)

        settings_and_output_layout.addWidget(QLabel("Output:"))

        settings_and_output_layout.addWidget(self.process_output)

        graph_layout = QVBoxLayout()
        graph_layout.addWidget(QLabel("Choose model:"))
        self.graph_model_combo_box = QComboBox()
        self.graph_model_combo_box.addItems(models)
        self.graph_model_selection = self.model_selection
        self.graph_model_combo_box.currentTextChanged.connect(self.graph_model_selection_changed)
        graph_layout.addWidget(self.graph_model_combo_box)

        graph_layout.addWidget(QLabel("Graph:"))
        self.graph: pg.PlotWidget = pg.PlotWidget()
        self.graph.setBackground('w')
        graph_layout.addWidget(self.graph)

        layout.addLayout(settings_and_output_layout)
        layout.addLayout(graph_layout)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def model_selection_changed(self, value):
        self.model_selection = value
        self.graph_model_combo_box.setCurrentText(self.model_selection)

    def pruning_sparsity_changed(self, value):
        self.pruning_sparsity = value

    def start_pruning(self):
        if self.process is None:
            env = QProcessEnvironment.systemEnvironment()
            env.insert("TQDM_POSITION", "-1")
            env.insert("HF_HUB_DISABLE_PROGRESS_BARS", "0")
            env.insert("PYTHONUNBUFFERED", "1")
            self.process = QProcess()
            self.process.setProcessEnvironment(env)

            self.process.readyReadStandardOutput.connect(self.handle_process_stdout)
            self.process.readyReadStandardError.connect(self.handle_process_stderr)
            self.process.finished.connect(self.process_finished)
            self.process.start("python3", [
                "../experiments/run_slicegpt.py",
                "--model",
                f"{self.model_selection}",
                "--save-dir",
                "../sliced/",
                "--sparsity",
                f"{self.pruning_sparsity}",
                "--no-wandb"
            ])

    def process_finished(self):
        self.process = None
        self.append_process_output("Finished.")
        model = self.models_dict[self.model_selection]
        file_path = model["file_path"]
        data_by_sparsity = model["data_by_sparsity"]
        with open(file_path, "w") as f:
            json.dump({
                "data_by_sparsity": data_by_sparsity,
            }, f)

    def handle_process_stdout(self):
        data = self.process.readAllStandardOutput().data().decode("utf-8", 'ignore')
        data = self.ansi_re.sub('', data)
        self.append_process_output(data)

    def handle_process_stderr(self):
        data = self.process.readAllStandardError().data().decode("utf-8", 'ignore')
        data = self.ansi_re.sub('', data)
        for name, pattern in self.metric_res.items():
            match = re.search(pattern, data)
            if match:
                if name in ["original_model_parameters", "sliced_model_parameters"]:
                    val = int(match.group(1).replace(",", ""))
                else:
                    val = round(float(match.group(1)), 4)
                print(name, val)

                try:
                    data_by_sparsity = self.models_dict[self.model_selection]["data_by_sparsity"]
                except KeyError:
                    self.models_dict[self.model_selection]["data_by_sparsity"] = {}
                    data_by_sparsity = self.models_dict[self.model_selection]["data_by_sparsity"]

                if name == "original_model_parameters":
                    sparsity_key_in_json = None
                    for sparsity_key in data_by_sparsity:
                        if float(sparsity_key) == 0:
                            sparsity_key_in_json = sparsity_key
                            break
                    if sparsity_key_in_json:
                        data_by_sparsity = data_by_sparsity[sparsity_key_in_json]
                    else:
                        data_by_sparsity[str(float(0))] = {}
                        data_by_sparsity = data_by_sparsity[str(float(0))]
                    data_by_sparsity['params_count'] = val
                    continue

                sparsity_key_in_json = None
                for sparsity_key in data_by_sparsity:
                    if float(sparsity_key) == self.pruning_sparsity:
                        sparsity_key_in_json = sparsity_key
                        break
                if sparsity_key_in_json:
                    data_by_sparsity = data_by_sparsity[sparsity_key_in_json]
                else:
                    data_by_sparsity[str(self.pruning_sparsity)] = {}
                    data_by_sparsity = data_by_sparsity[str(self.pruning_sparsity)]

                try:
                    data_by_sparsity["metrics_by_baseline"]
                except KeyError:
                    data_by_sparsity["metrics_by_baseline"] = {
                        "0.25": {
                            "arc_challenge": -1,
                            "arc_easy": -1,
                            "hellaswag": -1
                        },
                        "0.5": {
                            "piqa": -1,
                            "winogrande": -1
                        }
                    }
                if name == "perplexity":
                    data_by_sparsity[name] = val
                elif name == "sliced_model_parameters":
                    data_by_sparsity["params_count"] = val
                else:
                    for baseline in data_by_sparsity["metrics_by_baseline"]:
                        for metric in data_by_sparsity["metrics_by_baseline"][baseline]:
                            if metric == name:
                                data_by_sparsity["metrics_by_baseline"][baseline][name] = val

        self.append_process_output(data)

    def append_process_output(self, value):
        self.process_output.appendPlainText(value)

    def graph_model_selection_changed(self, value):
        self.graph_model_selection = value
        graph_data = self.get_graph_data_for_model(value, SPARSITY_X_FIELD, SPARSITY_X_FIELD,
                                                   [("arc_challenge", "arc_challenge")])
        print(graph_data)
        self.draw_graph(graph_data)

    def get_field_value_iter(self, dict_to_iterate: dict, field):
        for key, value in dict_to_iterate.items():
            if key == field:
                return value
            if isinstance(value, dict):
                return self.get_field_value_iter(value, field)

    def get_graph_data_for_model(self, model_name, x_field, x_label, y_list) -> list:
        res = []
        for y_field, y_label in y_list:
            try:
                model_data = self.models_dict[model_name]
                data_by_sparsity = model_data["data_by_sparsity"]
                x_values = []
                y_values = []
                if x_field == SPARSITY_X_FIELD:
                    for key, value in data_by_sparsity.items():
                        y_val = self.get_field_value_iter(value, y_field)
                        if y_val is not None:
                            x_values.append(float(key))
                            y_values.append(float(y_val))
                else:
                    for key, value in data_by_sparsity.items():
                        x_val = self.get_field_value_iter(value, x_field)
                        y_val = self.get_field_value_iter(value, y_field)
                        if x_val is not None and y_val is not None:
                            x_values.append(float(x_val))
                            y_values.append(float(y_val))
                x_values.sort()
                y_values.sort(reverse=True)
                x_new = np.linspace(min(x_values), max(x_values), 100)
                spl = make_interp_spline(x_values, y_values, k=3)
                y_new = spl(x_new)
                res.append({
                    "x": {
                        "label": x_label,
                        "values": x_new,
                    },
                    "y": {
                        "label": y_label,
                        "values": y_new,
                    }
                })
            except KeyError:
                res.append({
                    "x": {
                        "label": x_label,
                        "values": [],
                    },
                    "y": {
                        "label": y_label,
                        "values": [],
                    }
                })

        return res

    def draw_graph(self, data):
        self.graph.clear()
        pen = pg.mkPen(color=(255, 0, 0), width=5)
        for d in data:
            x_label, x_values = d["x"]['label'], d["x"]['values']
            y_label, y_values = d["y"]['label'], d["y"]['values']
            self.graph.setLabel("bottom", x_label)
            self.graph.setLabel("left", "Precision")
            self.graph.addLegend()
            self.graph.plot(x_values, y_values, pen=pen, symbol='o', name=y_label)

    def evaluate_perplexity(self):
        if self.process is None:
            env = QProcessEnvironment.systemEnvironment()
            env.insert("TQDM_POSITION", "-1")
            env.insert("HF_HUB_DISABLE_PROGRESS_BARS", "0")
            env.insert("PYTHONUNBUFFERED", "1")
            self.process = QProcess()
            self.process.setProcessEnvironment(env)

            self.process.readyReadStandardOutput.connect(self.handle_process_stdout)
            self.process.readyReadStandardError.connect(self.handle_process_stderr)
            self.process.finished.connect(self.process_finished)
            args = [
                "../experiments/run_slicegpt.py",
                "--model",
                f"{self.model_selection}",
                "--sparsity",
                f"{self.pruning_sparsity}",
                "--no-wandb",
                "--ppl-only"
            ]
            if self.pruning_sparsity:
                args.extend(["--sliced-model-path", "../sliced/"])

            self.process.start("python3", args)

    def evaluate_lm(self):
        if self.process is None:
            env = QProcessEnvironment.systemEnvironment()
            env.insert("TQDM_POSITION", "-1")
            env.insert("HF_HUB_DISABLE_PROGRESS_BARS", "0")
            env.insert("PYTHONUNBUFFERED", "1")
            self.process = QProcess()
            self.process.setProcessEnvironment(env)

            self.process.readyReadStandardOutput.connect(self.handle_process_stdout)
            self.process.readyReadStandardError.connect(self.handle_process_stderr)
            self.process.finished.connect(self.process_finished)
            args = [
                "../experiments/run_lm_eval.py",
                "--model",
                f"{self.model_selection}",
                "--no-wandb",
            ]
            if self.pruning_sparsity:
                args.extend(["--sparsity", f"{self.pruning_sparsity}", "--sliced-model-path", "../sliced/"])

            self.process.start("python3", args)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
