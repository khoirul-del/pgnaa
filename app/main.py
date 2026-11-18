import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


CHANNELS = 1024
DEFAULT_BAUD = 9600

FEATURE_COLUMNS = [
    "peak1_channel", "peak1_intensity", "peak1_fwhm", "peak1_area",
    "peak2_channel", "peak2_intensity", "peak2_fwhm", "peak2_area",
    "peak3_channel", "peak3_intensity", "peak3_fwhm", "peak3_area",
]


# PURE MENGGUNAKAN PYTHON RANDOM FOREST tapi butuh data training terlebih dahulu
# Tidak menggunakan scikit-learn / DLL _libsvm



def find_peaks_numpy(data, prominence=10.0, distance=12):
    """Deteksi puncak lokal sederhana tanpa SciPy.

    Prominence dihitung relatif terhadap minimum lokal di kiri/kanan
    sampai bertemu puncak yang lebih tinggi atau batas sinyal.
    Lebar dihitung pada setengah prominence (estimasi FWHM).
    """
    y = np.asarray(data, dtype=float).ravel()
    n = len(y)
    if n < 3:
        return np.array([], dtype=int), {"prominences": np.array([])}, np.array([])

    candidates = np.where(
        (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
    )[0] + 1

    found = []
    prominences = []
    widths = []
    min_distance = max(1, int(distance))
    min_prominence = max(0.0, float(prominence))

    for peak in candidates:
        height = y[peak]
        left = peak
        while left > 0 and y[left - 1] <= height:
            left -= 1
        right = peak
        while right < n - 1 and y[right + 1] <= height:
            right += 1

        left_min = float(np.min(y[left:peak + 1]))
        right_min = float(np.min(y[peak:right + 1]))
        prom = float(height - max(left_min, right_min))
        if prom < min_prominence:
            continue

        level = height - prom * 0.5
        l = peak
        while l > 0 and y[l] > level:
            l -= 1
        r = peak
        while r < n - 1 and y[r] > level:
            r += 1

        # Interpolasi sederhana untuk estimasi lebar sub-channel.
        left_cross = float(l)
        if l < peak and y[l + 1] != y[l]:
            left_cross += (level - y[l]) / (y[l + 1] - y[l])
        right_cross = float(r)
        if r > peak and y[r - 1] != y[r]:
            right_cross -= (level - y[r]) / (y[r - 1] - y[r])
        width = max(0.0, right_cross - left_cross)

        found.append(int(peak))
        prominences.append(prom)
        widths.append(width)

    # Prioritaskan puncak paling menonjol jika jaraknya terlalu dekat.
    order = sorted(range(len(found)), key=lambda i: prominences[i], reverse=True)
    kept = []
    for idx in order:
        if all(abs(found[idx] - found[j]) >= min_distance for j in kept):
            kept.append(idx)
    kept.sort(key=lambda i: found[i])

    peak_indices = np.array([found[i] for i in kept], dtype=int)
    peak_prominences = np.array([prominences[i] for i in kept], dtype=float)
    peak_widths = np.array([widths[i] for i in kept], dtype=float)
    return peak_indices, {"prominences": peak_prominences}, peak_widths


class SimpleDecisionTree:
    def __init__(self, max_depth=7, min_samples_split=3, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng or np.random.default_rng()
        self.root = None
        self.n_classes = 0

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.root = self._build(X, y, depth=0)
        return self

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes)
        p = counts / len(y)
        return 1.0 - np.sum(p * p)

    def _best_split(self, X, y, feature_indices):
        parent_gini = self._gini(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        n = len(y)
        if n < self.min_samples_split:
            return None, None

        for feature in feature_indices:
            values = X[:, feature]
            unique = np.unique(values)

            if len(unique) < 2:
                continue

            # Batasi kandidat threshold untuk dataset yang besar.
            if len(unique) > 32:
                qs = np.linspace(0.05, 0.95, 19)
                thresholds = np.unique(np.quantile(unique, qs))
            else:
                thresholds = (unique[:-1] + unique[1:]) / 2.0

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask

                nl = int(np.sum(left_mask))
                nr = n - nl

                if nl == 0 or nr == 0:
                    continue

                gain = parent_gini - (
                    (nl / n) * self._gini(y[left_mask])
                    + (nr / n) * self._gini(y[right_mask])
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)

        return best_feature, best_threshold

    def _leaf(self, y):
        counts = np.bincount(y, minlength=self.n_classes)
        return {
            "leaf": True,
            "class": int(np.argmax(counts)),
            "counts": counts.astype(int),
        }

    def _build(self, X, y, depth):
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) <= 1
        ):
            return self._leaf(y)

        n_features = X.shape[1]

        if self.max_features is None:
            k = max(1, int(np.sqrt(n_features)))
        else:
            k = min(n_features, int(self.max_features))

        feature_indices = self.rng.choice(
            n_features, size=k, replace=False
        )

        feature, threshold = self._best_split(
            X, y, feature_indices
        )

        if feature is None:
            return self._leaf(y)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return self._leaf(y)

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build(
                X[left_mask], y[left_mask], depth + 1
            ),
            "right": self._build(
                X[right_mask], y[right_mask], depth + 1
            ),
        }

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["class"], node["counts"]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])

        return self._predict_one(x, node["right"])

    def predict_one(self, x):
        return self._predict_one(x, self.root)[0]

    def probability_one(self, x):
        _, counts = self._predict_one(x, self.root)
        total = max(1, int(np.sum(counts)))
        return counts / total


class SimpleRandomForest:
    def __init__(
        self,
        n_estimators=80,
        max_depth=7,
        min_samples_split=3,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.classes_ = np.unique(y)
        self.trees = []

        n = len(X)

        for _ in range(self.n_estimators):
            sample_idx = self.rng.integers(0, n, size=n)

            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max(1, int(np.sqrt(X.shape[1]))),
                rng=np.random.default_rng(
                    int(self.rng.integers(0, 2**31 - 1))
                ),
            )

            tree.fit(X[sample_idx], y[sample_idx])
            self.trees.append(tree)

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        n_classes = len(self.classes_)
        result = np.zeros((len(X), n_classes), dtype=float)

        for row_idx, x in enumerate(X):
            votes = np.zeros(n_classes, dtype=float)

            for tree in self.trees:
                pred = tree.predict_one(x)
                if 0 <= pred < n_classes:
                    votes[pred] += 1

            if np.sum(votes) > 0:
                votes /= np.sum(votes)

            result[row_idx] = votes

        return result

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


# ============================================================
# METRICS + STRATIFIED SPLIT
# ============================================================

def stratified_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []

    for cls in np.unique(y):
        indices = np.where(y == cls)[0]
        indices = indices.copy()
        rng.shuffle(indices)

        n_test = max(1, int(round(len(indices) * test_size)))
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
    )


def classification_metrics(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for actual, predicted in zip(y_true, y_pred):
        cm[int(actual), int(predicted)] += 1

    accuracy = (
        float(np.sum(np.diag(cm)) / np.sum(cm))
        if np.sum(cm) else 0.0
    )

    precisions = []
    recalls = []
    f1s = []

    for cls in range(n_classes):
        tp = cm[cls, cls]
        fp = np.sum(cm[:, cls]) - tp
        fn = np.sum(cm[cls, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return (
        accuracy,
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(f1s)),
        cm,
    )


# ============================================================
# DESKTOP APPLICATION
# ============================================================

class GammaAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "PGNAA Gamma Analyzer V3.6 - BRIN Serial | Auto Analysis"
        )
        self.root.geometry("1500x820")
        self.root.minsize(1180, 700)
        self.root.configure(bg="#edf2f8")
        self._configure_visual_theme()

        self.ser = None
        self.running = False
        self.reader_thread = None
        self.simulation_mode = False

        self.raw_counts = np.zeros(
            CHANNELS, dtype=float
        )
        self.processed_counts = np.zeros(
            CHANNELS, dtype=float
        )

        self.peaks = pd.DataFrame(
            columns=[
                "Peak", "Channel", "Intensity",
                "Prominence", "FWHM", "Area"
            ]
        )

        self.smooth_window = tk.IntVar(value=9)
        self.prominence = tk.DoubleVar(value=10.0)
        self.distance = tk.IntVar(value=12)

        self.model = None
        self.model_classes = []
        self.training_info = "Model: belum dilatih"
        self.history = []
        self.last_prediction = None
        self.last_confidence = None
        self._last_auto_analysis_at = 0.0
        self._auto_analysis_interval = 1.0
        self._last_auto_spectrum_signature = None

        # Status variables remain available even though the old
        # "Identifikasi Unsur Otomatis" panel is intentionally hidden.
        # Without these variables, clicking Start raises AttributeError
        # before the serial reader thread can begin.
        self.auto_analysis_var = tk.StringVar(
            value="Siap. Klik Start untuk memulai akuisisi."
        )
        self.auto_prediction_var = tk.StringVar(
            value="Hasil identifikasi: menunggu akuisisi."
        )
        self.auto_peak_var = tk.StringVar(value="Peak terdeteksi: 0")

        self._build_ui()
        self.refresh_ports()
        self.update_plot()

        self.root.protocol(
            "WM_DELETE_WINDOW", self.on_close
        )

    def _configure_visual_theme(self):
        """Tema berwarna dengan widget ttk yang tetap ringan untuk Windows 7."""
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.option_add("*Font", ("Segoe UI", 9))
        style.configure(".", font=("Segoe UI", 9))
        style.configure("TFrame", background="#edf2f8")
        style.configure("TLabel", background="#edf2f8", foreground="#26364a")
        style.configure("TLabelframe", background="#edf2f8",
                        bordercolor="#cbd7e5", relief="solid")
        style.configure("TLabelframe.Label", background="#edf2f8",
                        foreground="#176b48", font=("Segoe UI", 9, "bold"))
        style.configure("TButton", padding=(10, 6), background="#e0eaf4",
                        foreground="#20354b", bordercolor="#c3d2e2")
        style.map("TButton",
                  background=[("active", "#c9def1"), ("pressed", "#b8d2e9")],
                  foreground=[("disabled", "#8997a6")])
        style.configure("Accent.TButton", background="#168447",
                        foreground="white", font=("Segoe UI", 9, "bold"))
        style.map("Accent.TButton",
                  background=[("active", "#116c39"), ("pressed", "#0d5a30")],
                  foreground=[("active", "white")])
        style.configure("TNotebook", background="#edf2f8", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8),
                        background="#dce6f0", foreground="#31465c")
        style.map("TNotebook.Tab",
                  background=[("selected", "#176b48"), ("active", "#c8dfd2")],
                  foreground=[("selected", "white"), ("active", "#173d2b")])
        style.configure("Treeview", background="white", fieldbackground="white",
                        foreground="#26364a", rowheight=25,
                        bordercolor="#d8e1eb")
        style.configure("Treeview.Heading", background="#dce8f2",
                        foreground="#23415a", font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#c9e7d5")],
                  foreground=[("selected", "#153d29")])
        style.configure("TCombobox", padding=4)
        style.configure("TSpinbox", padding=4)

    def _build_ui(self):
        # Header
        header = ttk.Frame(self.root, padding=(14, 10))
        header.pack(fill="x")

        header.configure(style="Header.TFrame")
        ttk.Style().configure("Header.TFrame", background="#145c42")
        ttk.Style().configure("HeaderTitle.TLabel", background="#145c42",
                              foreground="white", font=("Segoe UI", 19, "bold"))
        ttk.Style().configure("HeaderStatus.TLabel", background="#145c42",
                              foreground="#d7f5e4", font=("Segoe UI", 10, "bold"))
        ttk.Label(
            header,
            text="PGNAA GAMMA ANALYZER",
            style="HeaderTitle.TLabel",
        ).pack(side="left")

        self.status_var = tk.StringVar(value="Status: DISCONNECTED")
        ttk.Label(
            header,
            textvariable=self.status_var,
            style="HeaderStatus.TLabel",
        ).pack(side="right")

        # Connection
        conn = ttk.LabelFrame(self.root, text="  Koneksi Alat  ", padding=10)
        conn.pack(fill="x", padx=14, pady=(0, 7))

        ttk.Label(conn, text="COM Port").grid(row=0, column=0, padx=(4, 6))
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(
            conn, textvariable=self.port_var, width=18, state="readonly"
        )
        self.port_combo.grid(row=0, column=1, padx=4)

        ttk.Button(
            conn, text="Refresh", command=self.refresh_ports
        ).grid(row=0, column=2, padx=5)

        ttk.Label(conn, text="Baudrate").grid(row=0, column=3, padx=(20, 6))
        self.baud_var = tk.StringVar(value=str(DEFAULT_BAUD))
        ttk.Combobox(
            conn,
            textvariable=self.baud_var,
            values=["9600", "19200", "38400", "57600", "115200"],
            width=12,
            state="readonly",
        ).grid(row=0, column=4, padx=4)

        self.connect_btn = ttk.Button(
            conn, text="Connect", command=self.toggle_connection,
            style="Accent.TButton"
        )
        self.connect_btn.grid(row=0, column=5, padx=(20, 4))

        # Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        self.analysis_tab = ttk.Frame(self.notebook, padding=8)
        self.cacahan_tab = ttk.Frame(self.notebook, padding=8)
        self.ml_tab = ttk.Frame(self.notebook, padding=8)
        self.result_tab = ttk.Frame(self.notebook, padding=8)

        self.notebook.add(self.analysis_tab, text="  Analisis Spektrum  ")
        self.notebook.add(self.cacahan_tab, text="  Data Cacahan  ")
        self.notebook.add(self.ml_tab, text="  Machine Learning  ")
        self.notebook.add(self.result_tab, text="  Hasil & Riwayat  ")

        self._build_analysis_tab()
        self._build_cacahan_tab()
        self._build_ml_tab()
        self._build_result_tab()

    def _build_analysis_tab(self):
        # Acquisition
        acq = ttk.LabelFrame(
            self.analysis_tab, text="  Akuisisi & Data  ", padding=10
        )
        acq.pack(fill="x", pady=(0, 7))

        for label, command in [
            ("Start", self.start_acquisition),
            ("Stop", self.stop_acquisition),
            ("Reset", self.reset_spectrum),
            ("Simulasi", self.toggle_simulation),
            ("Upload CSV", self.load_csv),
            ("Simpan CSV", self.save_csv),
            ("Download Grafik", self.save_graph),
        ]:
            ttk.Button(
                acq, text=label, command=command, width=11,
                style="Accent.TButton" if label == "Start" else "TButton"
            ).pack(side="left", padx=4)

        stat_frame = ttk.Frame(acq)
        stat_frame.pack(side="right")

        self.channel_var = tk.StringVar(value="Peak Channel: -")
        self.max_var = tk.StringVar(value="Max Count: 0")
        self.total_var = tk.StringVar(value="Total Count: 0")
        self.peaks_var = tk.StringVar(value="Detected Peaks: 0")

        for var in [
            self.channel_var, self.max_var,
            self.total_var, self.peaks_var
        ]:
            ttk.Label(
                stat_frame,
                textvariable=var,
                font=("Segoe UI", 9, "bold"),
            ).pack(side="left", padx=8)

        # Analysis controls
        analysis = ttk.LabelFrame(
            self.analysis_tab, text="  Parameter Analisis  ", padding=10
        )
        analysis.pack(fill="x", pady=(0, 7))

        ttk.Label(analysis, text="Smoothing Window").pack(side="left", padx=(4, 5))
        ttk.Spinbox(
            analysis, from_=3, to=51, increment=2,
            textvariable=self.smooth_window, width=6
        ).pack(side="left", padx=3)

        ttk.Label(analysis, text="Prominence").pack(side="left", padx=(18, 5))
        ttk.Spinbox(
            analysis, from_=1, to=10000, increment=1,
            textvariable=self.prominence, width=8
        ).pack(side="left", padx=3)

        ttk.Label(analysis, text="Min Distance").pack(side="left", padx=(18, 5))
        ttk.Spinbox(
            analysis, from_=1, to=100, increment=1,
            textvariable=self.distance, width=6
        ).pack(side="left", padx=3)

        ttk.Button(
            analysis, text="Preprocessing",
            command=self.run_preprocessing, width=14
        ).pack(side="left", padx=(20, 4))

        ttk.Button(
            analysis, text="Deteksi Peak",
            command=self.detect_peaks, width=13
        ).pack(side="left", padx=4)

        ttk.Button(
            analysis, text="Reset Analisis",
            command=self.reset_analysis, width=13
        ).pack(side="left", padx=4)

        # Panel identifikasi otomatis disembunyikan sesuai permintaan UI.

        # Main analysis area
        main = ttk.Frame(self.analysis_tab)
        main.pack(fill="both", expand=True)

        plot_frame = ttk.LabelFrame(
            main, text="  Spektrum Gamma (1024 Channel)  ", padding=8
        )
        plot_frame.pack(side="left", fill="both", expand=True)

        # Tampilan spektrum bergaya MCA/NaI(Tl): histogram hijau terisi.
        self.figure = Figure(figsize=(13, 6.2), dpi=100, facecolor="#f4f7fb")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#fbfdff")
        self.ax.set_xlabel("Channel", fontsize=10, color="#176f78")
        self.ax.set_ylabel("Counts", fontsize=10, color="#176f78")
        self.ax.set_xlim(0, CHANNELS - 1)
        # Auto-scale on startup so low-count spectra remain visible.
        self.ax.set_ylim(0, 10)
        self.ax.grid(True, which="major", color="#222222", linestyle=":", linewidth=0.75, alpha=0.85)
        self.ax.set_axisbelow(True)
        self.ax.tick_params(axis="both", colors="#176f78", labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#c8d4e2")

        x = np.arange(CHANNELS)
        self.spectrum_fill = self.ax.fill_between(
            x, self.raw_counts, step="mid",
            color="#079da5", alpha=0.92, label="Spektrum cacahan"
        )
        self.line, = self.ax.plot(
            x, self.raw_counts, color="#078b91", linewidth=0.65,
            drawstyle="steps-mid", label="Data mentah"
        )
        self.processed_line, = self.ax.plot(
            x, self.processed_counts, color="#f0a52b",
            linewidth=1.35, alpha=0.95, label="Hasil preprocessing"
        )
        self.peak_scatter = self.ax.scatter(
            [], [], marker="v", s=42, color="#d94841",
            edgecolors="white", linewidths=0.5, zorder=5,
            label="Puncak terdeteksi"
        )
        self.ax.legend(
            loc="upper right", frameon=True, facecolor="white",
            edgecolor="#dce5ef", fontsize=8
        )
        self.figure.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.12)

        self.canvas = FigureCanvasTkAgg(
            self.figure, master=plot_frame
        )
        # Canvas mengikuti ruang layar yang tersedia; tidak memaksa tinggi
        # tetap yang dapat mendorong bagian bawah grafik keluar dari layar.
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill="x", side="bottom")
        self.plot_toolbar = NavigationToolbar2Tk(
            self.canvas, toolbar_frame, pack_toolbar=False
        )
        self.plot_toolbar.update()
        self.plot_toolbar.pack(side="left", fill="x")

        # Right tables
        # Panel tabel dibuat lebih ramping agar area grafik mendapat ruang maksimum.
        right = ttk.Frame(main, width=410)
        # Explicitly reserve the right column before the expanding plot.
        right.pack(side="right", fill="y", padx=(7, 0), before=plot_frame)
        right.pack_propagate(False)

        channel_frame = ttk.LabelFrame(
            right, text="  Data Channel + Count  ",
            padding=6, height=250
        )
        channel_frame.pack(fill="x", expand=False)
        channel_frame.pack_propagate(False)

        channel_body = ttk.Frame(channel_frame)
        channel_body.pack(fill="both", expand=True)

        self.table = ttk.Treeview(
            channel_body,
            columns=("channel", "count"),
            show="headings",
        )
        self.table.heading("channel", text="CHANNEL")
        self.table.heading("count", text="COUNT")
        self.table.column(
            "channel", width=95, anchor="center", stretch=False
        )
        self.table.column(
            "count", width=220, anchor="e", stretch=True
        )

        ch_scroll = ttk.Scrollbar(
            channel_body, orient="vertical", command=self.table.yview
        )
        self.table.configure(yscrollcommand=ch_scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        ch_scroll.pack(side="right", fill="y")

        peak_frame = ttk.LabelFrame(
            right, text="  Hasil Deteksi Peak  ", padding=7
        )
        peak_frame.pack(fill="both", expand=True, pady=(8, 0))

        peak_body = ttk.Frame(peak_frame)
        peak_body.pack(fill="both", expand=True)

        cols = ("peak", "channel", "intensity", "fwhm", "area")
        self.peak_table = ttk.Treeview(
            peak_body, columns=cols, show="headings"
        )

        headers = {
            "peak": "PEAK",
            "channel": "CHANNEL",
            "intensity": "INTENSITY",
            "fwhm": "FWHM",
            "area": "AREA",
        }
        widths = {
            "peak": 48, "channel": 68,
            "intensity": 88, "fwhm": 62, "area": 88
        }

        for col in cols:
            self.peak_table.heading(col, text=headers[col])
            self.peak_table.column(
                col,
                width=widths[col],
                anchor="center" if col in ("peak", "channel") else "e",
                stretch=True,
            )

        peak_scroll = ttk.Scrollbar(
            peak_body, orient="vertical", command=self.peak_table.yview
        )
        self.peak_table.configure(yscrollcommand=peak_scroll.set)
        self.peak_table.pack(side="left", fill="both", expand=True)
        peak_scroll.pack(side="right", fill="y")

        self._refresh_channel_table()
        self._refresh_peak_table()


    def _build_cacahan_tab(self):
        summary = ttk.LabelFrame(
            self.cacahan_tab,
            text="  Ringkasan Data Cacahan  ",
            padding=12
        )
        summary.pack(fill="x", pady=(0, 8))

        self.cacah_total_var = tk.StringVar(value="Total Cacahan: 0")
        self.cacah_max_var = tk.StringVar(value="Cacahan Tertinggi: 0")
        self.cacah_channel_var = tk.StringVar(value="Channel Tertinggi: -")
        self.cacah_avg_var = tk.StringVar(value="Rata-rata Cacahan: 0.00")

        for i, var in enumerate([
            self.cacah_total_var,
            self.cacah_max_var,
            self.cacah_channel_var,
            self.cacah_avg_var
        ]):
            card = ttk.Frame(
                summary, relief="ridge", borderwidth=1, padding=10
            )
            card.grid(row=0, column=i, padx=5, sticky="ew")
            ttk.Label(
                card, textvariable=var,
                font=("Segoe UI", 10, "bold")
            ).pack()
            summary.columnconfigure(i, weight=1)

        data_frame = ttk.LabelFrame(
            self.cacahan_tab,
            text="  Data Cacahan Spektrum - 1024 Channel  ",
            padding=8
        )
        data_frame.pack(fill="both", expand=True)

        top = ttk.Frame(data_frame)
        top.pack(fill="x", pady=(0, 7))

        ttk.Label(
            top,
            text="Data mentah hasil akuisisi: Channel 0-1023 dan Count/Cacahan."
        ).pack(side="left")

        ttk.Button(
            top, text="Refresh Data",
            command=self._refresh_cacahan_table
        ).pack(side="right", padx=4)

        ttk.Button(
            top, text="Simpan Cacahan CSV",
            command=self.save_cacahan_csv
        ).pack(side="right", padx=4)

        body = ttk.Frame(data_frame)
        body.pack(fill="both", expand=True)

        self.cacahan_table = ttk.Treeview(
            body,
            columns=("no", "channel", "count"),
            show="headings"
        )
        self.cacahan_table.heading("no", text="NO")
        self.cacahan_table.heading("channel", text="CHANNEL")
        self.cacahan_table.heading("count", text="COUNT / CACAHAN")

        self.cacahan_table.column(
            "no", width=80, anchor="center", stretch=False
        )
        self.cacahan_table.column(
            "channel", width=160, anchor="center", stretch=False
        )
        self.cacahan_table.column(
            "count", width=260, anchor="e", stretch=True
        )

        scroll = ttk.Scrollbar(
            body, orient="vertical",
            command=self.cacahan_table.yview
        )
        self.cacahan_table.configure(yscrollcommand=scroll.set)
        self.cacahan_table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self._refresh_cacahan_table()

    def _update_cacahan_summary(self):
        if not hasattr(self, "cacah_total_var"):
            return

        total = int(np.sum(self.raw_counts))
        max_count = float(np.max(self.raw_counts))
        peak_channel = int(np.argmax(self.raw_counts)) if max_count > 0 else "-"
        average = float(np.mean(self.raw_counts))

        self.cacah_total_var.set(f"Total Cacahan: {total:,}")
        self.cacah_max_var.set(f"Cacahan Tertinggi: {int(max_count):,}")
        self.cacah_channel_var.set(f"Channel Tertinggi: {peak_channel}")
        self.cacah_avg_var.set(f"Rata-rata Cacahan: {average:.2f}")

    def _refresh_cacahan_table(self):
        if not hasattr(self, "cacahan_table"):
            return

        for item in self.cacahan_table.get_children():
            self.cacahan_table.delete(item)

        for ch in range(CHANNELS):
            self.cacahan_table.insert(
                "", "end",
                values=(ch + 1, ch, f"{int(self.raw_counts[ch]):,}")
            )

        self._update_cacahan_summary()

    def save_cacahan_csv(self):
        path = filedialog.asksaveasfilename(
            title="Simpan Data Cacahan",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["channel", "count"])
                for ch, count in enumerate(self.raw_counts):
                    writer.writerow([ch, int(count)])

            messagebox.showinfo(
                "Data Cacahan",
                "1024 data Channel + Count berhasil disimpan."
            )
        except Exception as e:
            messagebox.showerror("Gagal Menyimpan Cacahan", str(e))

    def _build_result_tab(self):
        summary = ttk.LabelFrame(
            self.result_tab,
            text="  Hasil Identifikasi  ",
            padding=14
        )
        summary.pack(fill="x", pady=(0, 8))

        self.result_element_var = tk.StringVar(value="Unsur: -")
        self.result_confidence_var = tk.StringVar(value="Confidence: -")
        self.result_total_var = tk.StringVar(value="Total Cacahan: 0")
        self.result_peaks_var = tk.StringVar(value="Detected Peaks: 0")

        for i, var in enumerate([
            self.result_element_var,
            self.result_confidence_var,
            self.result_total_var,
            self.result_peaks_var
        ]):
            card = ttk.Frame(
                summary, relief="ridge", borderwidth=1, padding=10
            )
            card.grid(row=0, column=i, padx=5, sticky="ew")
            ttk.Label(
                card, textvariable=var,
                font=("Segoe UI", 11, "bold")
            ).pack()
            summary.columnconfigure(i, weight=1)

        actions = ttk.LabelFrame(
            self.result_tab,
            text="  Aksi Hasil  ",
            padding=10
        )
        actions.pack(fill="x", pady=(0, 8))

        ttk.Button(
            actions, text="Simpan ke Riwayat",
            command=self.save_history, width=18
        ).pack(side="left", padx=5)

        ttk.Button(
            actions, text="Simpan Hasil CSV",
            command=self.save_result_csv, width=18
        ).pack(side="left", padx=5)

        ttk.Button(
            actions, text="Export PDF",
            command=self.export_result_pdf, width=15
        ).pack(side="left", padx=5)

        ttk.Button(
            actions, text="Hapus Riwayat",
            command=self.clear_history, width=15
        ).pack(side="left", padx=5)

        history_frame = ttk.LabelFrame(
            self.result_tab,
            text="  Riwayat Identifikasi  ",
            padding=8
        )
        history_frame.pack(fill="both", expand=True)

        cols = ("time", "element", "confidence", "peaks", "total")
        self.history_table = ttk.Treeview(
            history_frame, columns=cols, show="headings"
        )

        headers = {
            "time": "WAKTU",
            "element": "UNSUR",
            "confidence": "CONFIDENCE",
            "peaks": "PEAK",
            "total": "TOTAL CACAHAN"
        }
        widths = {
            "time": 170, "element": 120,
            "confidence": 130, "peaks": 90,
            "total": 160
        }

        for col in cols:
            self.history_table.heading(col, text=headers[col])
            self.history_table.column(
                col, width=widths[col], anchor="center"
            )

        scroll = ttk.Scrollbar(
            history_frame, orient="vertical",
            command=self.history_table.yview
        )
        self.history_table.configure(yscrollcommand=scroll.set)
        self.history_table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _refresh_result_tab(self):
        element = self.last_prediction
        confidence = self.last_confidence

        self.result_element_var.set(
            f"Unsur: {element if element else '-'}"
        )
        self.result_confidence_var.set(
            f"Confidence: {confidence:.2%}"
            if confidence is not None else "Confidence: -"
        )
        self.result_total_var.set(
            f"Total Cacahan: {int(np.sum(self.raw_counts)):,}"
        )
        self.result_peaks_var.set(
            f"Detected Peaks: {len(self.peaks)}"
        )

    def _refresh_history_table(self):
        for item in self.history_table.get_children():
            self.history_table.delete(item)

        for item in self.history:
            self.history_table.insert(
                "", "end",
                values=(
                    item["time"],
                    item["element"],
                    item["confidence"],
                    item["peaks"],
                    f"{item['total']:,}"
                )
            )

    def save_history(self):
        if not self.last_prediction:
            messagebox.showwarning(
                "Riwayat",
                "Belum ada hasil identifikasi."
            )
            return

        item = {
            "time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "element": str(self.last_prediction),
            "confidence": (
                f"{self.last_confidence:.2%}"
                if self.last_confidence is not None else "-"
            ),
            "peaks": len(self.peaks),
            "total": int(np.sum(self.raw_counts))
        }

        self.history.append(item)
        self._refresh_history_table()
        self.status_var.set("Status: RIWAYAT DISIMPAN")

    def clear_history(self):
        if not self.history:
            return
        if messagebox.askyesno(
            "Hapus Riwayat",
            "Hapus seluruh riwayat identifikasi?"
        ):
            self.history.clear()
            self._refresh_history_table()

    def save_result_csv(self):
        if self.peaks.empty:
            messagebox.showwarning(
                "Hasil CSV",
                "Belum ada hasil peak."
            )
            return

        path = filedialog.asksaveasfilename(
            title="Simpan Hasil Analisis",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "prediction", "confidence", "total_count",
                    "peak", "channel", "intensity", "fwhm", "area"
                ])

                for _, row in self.peaks.iterrows():
                    writer.writerow([
                        self.last_prediction or "",
                        (
                            f"{self.last_confidence:.4f}"
                            if self.last_confidence is not None else ""
                        ),
                        int(np.sum(self.raw_counts)),
                        int(row["Peak"]),
                        int(row["Channel"]),
                        f"{row['Intensity']:.4f}",
                        f"{row['FWHM']:.4f}",
                        f"{row['Area']:.4f}"
                    ])

            messagebox.showinfo(
                "Hasil CSV",
                "Hasil analisis berhasil disimpan."
            )
        except Exception as e:
            messagebox.showerror("Export CSV Gagal", str(e))

    def export_result_pdf(self):
        if self.peaks.empty:
            messagebox.showwarning(
                "PDF",
                "Belum ada hasil peak."
            )
            return

        path = filedialog.asksaveasfilename(
            title="Export Laporan Analisis",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")]
        )
        if not path:
            return

        try:
            styles = getSampleStyleSheet()
            doc = SimpleDocTemplate(
                path, pagesize=A4,
                leftMargin=36, rightMargin=36,
                topMargin=36, bottomMargin=36
            )

            story = [
                Paragraph(
                    "PGNAA GAMMA ANALYZER - LAPORAN HASIL ANALISIS",
                    styles["Title"]
                ),
                Spacer(1, 10),
                Paragraph(
                    f"Waktu: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
                    styles["Normal"]
                ),
                Paragraph(
                    f"Hasil Identifikasi: <b>{self.last_prediction or '-'}</b>",
                    styles["Normal"]
                ),
                Paragraph(
                    (
                        f"Confidence: <b>{self.last_confidence:.2%}</b>"
                        if self.last_confidence is not None
                        else "Confidence: -"
                    ),
                    styles["Normal"]
                ),
                Paragraph(
                    f"Total Cacahan: {int(np.sum(self.raw_counts)):,}",
                    styles["Normal"]
                ),
                Paragraph(
                    f"Jumlah Peak: {len(self.peaks)}",
                    styles["Normal"]
                ),
                Spacer(1, 12)
            ]

            data = [["Peak", "Channel", "Intensity", "FWHM", "Area"]]
            for _, row in self.peaks.iterrows():
                data.append([
                    str(int(row["Peak"])),
                    str(int(row["Channel"])),
                    f"{row['Intensity']:.2f}",
                    f"{row['FWHM']:.2f}",
                    f"{row['Area']:.2f}"
                ])

            table = Table(data, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE", (0,0), (-1,-1), 9)
            ]))
            story.append(table)
            doc.build(story)

            messagebox.showinfo(
                "PDF",
                "Laporan PDF berhasil dibuat."
            )
        except Exception as e:
            messagebox.showerror("Export PDF Gagal", str(e))

    def _build_ml_tab(self):
        top = ttk.Frame(self.ml_tab)
        top.pack(fill="x")

        dataset_frame = ttk.LabelFrame(
            top, text="  Training Model  ", padding=12
        )
        dataset_frame.pack(fill="x", pady=(0, 8))

        ttk.Label(
            dataset_frame,
            text="Random Forest",
            font=("Segoe UI", 13, "bold")
        ).grid(row=0, column=0, sticky="w", padx=5, pady=(0, 8))

        ttk.Label(
            dataset_frame,
            text="Dataset digunakan untuk melatih model identifikasi unsur."
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=5, pady=3)

        ttk.Button(
            dataset_frame,
            text="Upload Dataset",
            command=self.train_from_dataset,
            width=18
        ).grid(row=2, column=0, padx=5, pady=10, sticky="w")

        ttk.Button(
            dataset_frame,
            text="Identifikasi Unsur",
            command=self.identify_element,
            width=18
        ).grid(row=2, column=1, padx=5, pady=10, sticky="w")

        ttk.Button(
            dataset_frame,
            text="Reset Model",
            command=self.reset_model,
            width=15
        ).grid(row=2, column=2, padx=5, pady=10, sticky="w")

        self.model_var = tk.StringVar(value=self.training_info)
        ttk.Label(
            dataset_frame,
            textvariable=self.model_var,
            font=("Segoe UI", 9, "bold")
        ).grid(row=2, column=3, padx=12, pady=10, sticky="e")

        # Result cards
        metrics = ttk.LabelFrame(
            self.ml_tab, text="  Hasil Evaluasi Model  ", padding=12
        )
        metrics.pack(fill="x", pady=(0, 8))

        self.ml_accuracy_var = tk.StringVar(value="Accuracy: -")
        self.ml_precision_var = tk.StringVar(value="Precision: -")
        self.ml_recall_var = tk.StringVar(value="Recall: -")
        self.ml_f1_var = tk.StringVar(value="F1-Score: -")
        self.ml_data_var = tk.StringVar(value="Data: -")
        self.ml_class_var = tk.StringVar(value="Kelas: -")

        metric_vars = [
            self.ml_accuracy_var,
            self.ml_precision_var,
            self.ml_recall_var,
            self.ml_f1_var,
            self.ml_data_var,
            self.ml_class_var,
        ]

        for i, var in enumerate(metric_vars):
            card = ttk.Frame(metrics, relief="ridge", borderwidth=1, padding=10)
            card.grid(
                row=0 if i < 4 else 1,
                column=i if i < 4 else i - 4,
                padx=5, pady=5, sticky="ew"
            )
            ttk.Label(
                card,
                textvariable=var,
                font=("Segoe UI", 10, "bold")
            ).pack()
            metrics.columnconfigure(i if i < 4 else i - 4, weight=1)

        # Prediction
        result = ttk.LabelFrame(
            self.ml_tab, text="  Hasil Identifikasi Unsur  ", padding=14
        )
        result.pack(fill="x", pady=(0, 8))

        self.prediction_var = tk.StringVar(value="Prediksi: -")
        ttk.Label(
            result,
            textvariable=self.prediction_var,
            font=("Segoe UI", 16, "bold")
        ).pack(pady=8)

        ttk.Label(
            result,
            text="Gunakan tab Analisis Spektrum untuk menyiapkan spektrum, "
                 "kemudian kembali ke tab ini untuk identifikasi."
        ).pack()

        # Confusion matrix
        cm_frame = ttk.LabelFrame(
            self.ml_tab, text="  Confusion Matrix  ", padding=10
        )
        cm_frame.pack(fill="both", expand=True)

        self.cm_tree = ttk.Treeview(
            cm_frame, show="tree"
        )
        cm_scroll = ttk.Scrollbar(
            cm_frame, orient="vertical",
            command=self.cm_tree.yview
        )
        self.cm_tree.configure(yscrollcommand=cm_scroll.set)
        self.cm_tree.pack(side="left", fill="both", expand=True)
        cm_scroll.pack(side="right", fill="y")

        ttk.Label(
            self.ml_tab,
            text="Catatan: metrik hanya menggambarkan dataset yang dipilih; "
                 "bukan otomatis menjadi hasil penelitian final."
        ).pack(anchor="w", pady=(7, 0))

    # ---------------- Connection ----------------

    def refresh_ports(self):
        ports = [
            p.device
            for p in serial.tools.list_ports.comports()
        ]

        self.port_combo["values"] = ports

        if (
            ports
            and self.port_var.get() not in ports
        ):
            self.port_var.set(
                ports[0]
            )

    def toggle_connection(self):
        if (
            self.ser
            and self.ser.is_open
        ):
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        port = self.port_var.get()

        if not port:
            messagebox.showwarning(
                "Koneksi",
                "Pilih COM Port terlebih dahulu.",
            )
            return

        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=int(
                    self.baud_var.get()
                ),
                timeout=0.1,
            )

            self.status_var.set(
                f"Status: CONNECTED - {port}"
            )

            self.connect_btn.config(
                text="Disconnect"
            )

        except Exception as e:
            self.ser = None

            messagebox.showerror(
                "Koneksi Gagal",
                str(e)
            )

    def disconnect(self):
        self.running = False

        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass

        self.ser = None

        self.status_var.set(
            "Status: DISCONNECTED"
        )

        self.connect_btn.config(
            text="Connect"
        )

    # ---------------- Acquisition ----------------

    def start_acquisition(self):
        if self.simulation_mode:
            self.running = True
            self._last_auto_analysis_at = 0.0
            self.status_var.set("Status: SIMULATION | AUTO ANALYSIS")
            self.auto_analysis_var.set("Akuisisi simulasi berjalan; analisis otomatis aktif.")
            return

        if (
            not self.ser
            or not self.ser.is_open
        ):
            messagebox.showwarning(
                "Akuisisi",
                "Hubungkan aplikasi ke alat terlebih dahulu.",
            )

            return

        if self.running:
            return

        self.running = True
        self._last_auto_analysis_at = 0.0
        self.auto_analysis_var.set(
            "Akuisisi berjalan; preprocessing dan deteksi peak otomatis aktif."
        )
        self.status_var.set("Status: ACQUIRING - {}".format(self.port_var.get()))

        self.reader_thread = threading.Thread(
            target=self.serial_reader,
            daemon=True,
        )

        self.reader_thread.start()

        self.status_var.set(
            "Status: ACQUIRING"
        )

    def stop_acquisition(self):
        self.running = False
        self.auto_analysis_var.set("Akuisisi dihentikan. Hasil analisis terakhir dipertahankan.")

        if self.simulation_mode:
            self.status_var.set(
                "Status: SIMULATION READY"
            )

        elif (
            self.ser
            and self.ser.is_open
        ):
            self.status_var.set(
                f"Status: CONNECTED - {self.port_var.get()}"
            )

    def toggle_simulation(self):
        self.simulation_mode = (
            not self.simulation_mode
        )

        self.running = (
            self.simulation_mode
        )

        self.status_var.set(
            "Status: SIMULATION"
            if self.simulation_mode
            else "Status: DISCONNECTED"
        )

    def reset_spectrum(self):
        self.raw_counts[:] = 0
        self.processed_counts[:] = 0

        self.peaks = (
            self.peaks.iloc[0:0]
        )

        self._refresh_channel_table()
        self._refresh_peak_table()
        self._refresh_cacahan_table()
        self._update_peak_labels()
        self.last_prediction = None
        self.last_confidence = None
        if hasattr(self, "auto_prediction_var"):
            self.auto_prediction_var.set("Hasil identifikasi: menunggu akuisisi.")
            self.auto_peak_var.set("Peak terdeteksi: 0")
            self.auto_analysis_var.set("Spektrum direset. Klik Start untuk analisis otomatis.")
        self._refresh_result_tab()

        self.prediction_var.set(
            "Prediksi: -"
        )

        self.update_statistics()
        self.canvas.draw_idle()

    def serial_reader(self):
        while (
            self.running
            and self.ser
            and self.ser.is_open
        ):
            try:
                raw = self.ser.readline()

                if not raw:
                    continue

                text = raw.decode(
                    errors="ignore"
                ).strip()

                if not text:
                    continue

                # Protokol MCA BRIN: satu baris berisi nomor channel/event.
                # Setiap event menambah cacahan channel terkait sebanyak 1.
                # Format pasangan channel,count tetap didukung untuk sumber lain.
                if text.isdigit():
                    channel = int(text)
                    if 0 <= channel < CHANNELS:
                        self.raw_counts[channel] += 1.0
                    continue

                normalized = (
                    text
                    .replace(";", ",")
                    .replace("\\t", ",")
                )
                parts = (
                    normalized.split(",")
                    if "," in normalized
                    else normalized.split()
                )

                if len(parts) >= 2:
                    channel = int(float(parts[0]))
                    count = float(parts[1])
                    if 0 <= channel < CHANNELS:
                        self.raw_counts[channel] = count

            except (
                ValueError,
                UnicodeDecodeError
            ):
                continue

            except Exception as exc:
                self.running = False
                error_text = str(exc)
                self.root.after(
                    0,
                    lambda msg=error_text: self.status_var.set(
                        "Status: SERIAL ERROR - {}".format(msg[:90])
                    )
                )
                break

    def generate_simulated_spectrum(self):
        x = np.arange(
            CHANNELS,
            dtype=float
        )

        y = np.random.poisson(
            2,
            CHANNELS
        ).astype(float)

        simulated_peaks = [
            (145, 120, 10),
            (310, 240, 14),
            (512, 420, 18),
            (735, 180, 12),
            (890, 300, 16),
        ]

        for (
            center,
            amplitude,
            width
        ) in simulated_peaks:

            y += (
                amplitude
                * np.exp(
                    -0.5
                    * (
                        (x - center)
                        / width
                    ) ** 2
                )
            )

        y += np.random.normal(
            0,
            1.5,
            CHANNELS
        )

        self.raw_counts = np.clip(
            y,
            0,
            None
        )

    # ---------------- Analysis ----------------

    def _remove_isolated_spikes(self, counts):
        """Suppress isolated one-channel impulses without clipping real broad peaks.

        The raw acquisition array is preserved. A bin is treated as a display/
        analysis spike only when it exceeds 12.5 counts AND is sharply isolated
        from both immediate shoulders and the wider local neighborhood.
        """
        raw = np.asarray(counts, dtype=float)
        cleaned = raw.copy()
        if raw.size < 9:
            return cleaned

        for i in range(3, len(raw) - 3):
            left = float(raw[i - 1])
            right = float(raw[i + 1])
            shoulder = max(left, right)
            neighborhood = np.r_[raw[i - 3:i], raw[i + 1:i + 4]]
            local_median = float(np.median(neighborhood))
            local_mad = float(np.median(np.abs(neighborhood - local_median)))
            robust_noise = 1.4826 * local_mad

            # Do not remove merely high counts: require a narrow, unsupported
            # impulse with weak adjacent channels and a clear local contrast.
            isolated = (
                raw[i] > 12.5
                and raw[i] > max(shoulder * 3.0,
                                local_median + max(8.0, 4.0 * robust_noise))
                and (left < raw[i] * 0.30)
                and (right < raw[i] * 0.30)
            )
            if isolated:
                cleaned[i] = 0.5 * (left + right)

        return cleaned

    def _make_analysis_spectrum(self, counts):
        """Spike filtering, moving-average smoothing, dan baseline correction."""
        cleaned = self._remove_isolated_spikes(counts)
        window = max(3, int(self.smooth_window.get()))
        if window % 2 == 0:
            window += 1
        window = min(window, 51)
        kernel = np.ones(window, dtype=float) / float(window)
        smoothed = np.convolve(cleaned, kernel, mode="same")

        x = np.arange(CHANNELS, dtype=float)
        # Estimasi baseline polinomial berderajat rendah seperti alur V3.6.
        coeff = np.polyfit(x, smoothed, 2)
        baseline = np.polyval(coeff, x)
        corrected = np.clip(smoothed - baseline, 0.0, None)
        return cleaned, corrected

    def run_preprocessing(self):
        if np.max(self.raw_counts) <= 0:
            messagebox.showwarning("Preprocessing", "Belum ada data spektrum.")
            return
        cleaned, corrected = self._make_analysis_spectrum(self.raw_counts)
        self.processed_counts = corrected
        self.auto_analysis_var.set(
            "Preprocessing selesai: isolated-spike filter + smoothing + baseline correction."
        )
        self.canvas.draw_idle()

    def detect_peaks(self):
        if np.max(
            self.processed_counts
        ) > 0:

            data = (
                self.processed_counts
            )

        else:
            data = self._make_analysis_spectrum(self.raw_counts)[1]

        if np.max(data) <= 0:
            if not silent:
                messagebox.showwarning("Deteksi Peak", "Belum ada data spektrum.")
            self.peaks = self.peaks.iloc[0:0]
            self._refresh_peak_table()
            self._update_peak_labels()
            return

        indices, properties, widths = find_peaks_numpy(
            data,
            prominence=float(self.prominence.get()),
            distance=int(self.distance.get()),
        )

        if len(indices) == 0:
            self.peaks = (
                self.peaks.iloc[0:0]
            )

            self._refresh_peak_table()
            self._update_peak_labels()

            if not silent:
                messagebox.showinfo("Deteksi Peak", "Tidak ditemukan peak.")
            return

        rows = []

        for i, idx in enumerate(
            indices
        ):
            width = max(
                2.0,
                float(widths[i])
                * 2
            )

            left = max(
                0,
                int(idx - width)
            )

            right = min(
                CHANNELS - 1,
                int(idx + width)
            )

            rows.append({
                "Peak": i + 1,
                "Channel": int(idx),
                "Intensity": float(
                    data[idx]
                ),
                "Prominence": float(
                    properties[
                        "prominences"
                    ][i]
                ),
                "FWHM": float(
                    widths[i]
                ),
                "Area": float(
                    np.trapz(
                        data[
                            left:right + 1
                        ]
                    )
                ),
            })

        self.peaks = pd.DataFrame(
            rows
        )

        self._refresh_peak_table()
        self._update_peak_labels()

        self.peak_scatter.set_offsets(
            np.column_stack(
                (
                    self.peaks[
                        "Channel"
                    ],
                    self.peaks[
                        "Intensity"
                    ],
                )
            )
        )

        self.status_var.set(
            "Status: PEAK DETECTION "
            f"SELESAI | {len(self.peaks)} peak"
        )

        self.canvas.draw_idle()

    def _run_automatic_analysis(self):
        """Jalankan analisis spektrum dan prediksi model secara periodik saat Start aktif."""
        if not self.running:
            return
        now = datetime.now().timestamp()
        if now - self._last_auto_analysis_at < self._auto_analysis_interval:
            return
        self._last_auto_analysis_at = now
        if np.max(self.raw_counts) <= 0:
            self.auto_analysis_var.set("Menunggu data cacahan dari perangkat...")
            return

        try:
            cleaned, corrected = self._make_analysis_spectrum(self.raw_counts)
            self.processed_counts = corrected
            # Peak dicari pada spektrum yang sudah dibersihkan, bukan pada spike mentah.
            indices, properties, widths = find_peaks_numpy(
                corrected,
                prominence=float(self.prominence.get()),
                distance=int(self.distance.get()),
            )
            rows = []
            for i, idx in enumerate(indices):
                width = max(2.0, float(widths[i]) * 2.0)
                left = max(0, int(idx - width))
                right = min(CHANNELS - 1, int(idx + width))
                rows.append({
                    "Peak": i + 1,
                    "Channel": int(idx),
                    "Intensity": float(corrected[idx]),
                    "Prominence": float(properties["prominences"][i]),
                    "FWHM": float(widths[i]),
                    "Area": float(np.trapz(corrected[left:right + 1])),
                })
            self.peaks = pd.DataFrame(
                rows,
                columns=["Peak", "Channel", "Intensity", "Prominence", "FWHM", "Area"]
            )
            self._refresh_peak_table()
            self._update_peak_labels()
            if len(self.peaks):
                self.peak_scatter.set_offsets(
                    np.column_stack((self.peaks["Channel"], self.peaks["Intensity"]))
                )
            else:
                self.peak_scatter.set_offsets(np.empty((0, 2)))

            self.auto_peak_var.set("Peak terdeteksi: {}".format(len(self.peaks)))
            if self.model is None:
                self.auto_prediction_var.set(
                    "Hasil identifikasi: model belum dilatih (upload dataset berlabel di tab Machine Learning)."
                )
                self.auto_analysis_var.set(
                    "LIVE: noise satu-channel terisolasi difilter; peak dianalisis otomatis."
                )
            elif self.peaks.empty:
                self.last_prediction = None
                self.last_confidence = None
                self.auto_prediction_var.set("Hasil identifikasi: belum ada peak yang memenuhi parameter.")
                self.auto_analysis_var.set("LIVE: menunggu peak yang valid.")
            else:
                features = self.build_feature_vector()
                if features is not None:
                    probabilities = self.model.predict_proba([features])[0]
                    best_index = int(np.argmax(probabilities))
                    prediction = self.model_classes[best_index]
                    confidence = float(probabilities[best_index])
                    self.last_prediction = prediction
                    self.last_confidence = confidence
                    self.prediction_var.set(
                        "Prediksi: {} | Confidence: {:.2%}".format(prediction, confidence)
                    )
                    self.auto_prediction_var.set(
                        "Kelas paling mungkin: {} | Confidence: {:.1%}".format(
                            prediction, confidence
                        )
                    )
                    self.auto_analysis_var.set(
                        "LIVE: identifikasi otomatis diperbarui {}".format(
                            datetime.now().strftime("%H:%M:%S")
                        )
                    )
                    self._refresh_result_tab()
        except Exception as exc:
            self.auto_analysis_var.set("Analisis otomatis gagal: {}".format(exc))

    def reset_analysis(self):
        self.processed_counts[:] = 0

        self.peaks = (
            self.peaks.iloc[0:0]
        )

        self.peak_scatter.set_offsets(
            np.empty((0, 2))
        )

        self._refresh_peak_table()
        self._update_peak_labels()

        self.prediction_var.set(
            "Prediksi: -"
        )

        self.status_var.set(
            "Status: ANALISIS DIRESET"
        )

        self.canvas.draw_idle()

    # ---------------- Feature extraction ----------------

    def build_feature_vector(self):
        if self.peaks.empty:
            self.detect_peaks()

        if self.peaks.empty:
            return None

        selected = (
            self.peaks
            .sort_values(
                "Intensity",
                ascending=False
            )
            .head(3)
            .sort_values("Channel")
        )

        values = []

        for _, row in selected.iterrows():
            values.extend([
                float(row["Channel"]),
                float(row["Intensity"]),
                float(row["FWHM"]),
                float(row["Area"]),
            ])

        while len(values) < len(
            FEATURE_COLUMNS
        ):
            values.append(0.0)

        return np.array(
            values,
            dtype=float
        )

    # ---------------- Machine Learning ----------------

    def train_from_dataset(self):
        path = filedialog.askopenfilename(
            title="Pilih Dataset Training",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        try:
            df = pd.read_csv(
                path
            )

            label_candidates = [
                "label",
                "element",
                "unsur",
                "class",
                "target",
            ]

            label_col = next(
                (
                    c
                    for c in df.columns
                    if str(c).strip().lower()
                    in label_candidates
                ),
                None,
            )

            if label_col is None:
                raise ValueError(
                    "Dataset harus memiliki "
                    "kolom label/element/unsur/class/target."
                )

            missing = [
                c
                for c in FEATURE_COLUMNS
                if c not in df.columns
            ]

            if missing:
                raise ValueError(
                    "Kolom fitur dataset belum lengkap:\n\n"
                    + ", ".join(missing)
                )

            clean = df.dropna(
                subset=FEATURE_COLUMNS
                + [label_col]
            ).copy()

            X = clean[
                FEATURE_COLUMNS
            ].astype(float).to_numpy()

            labels = (
                clean[label_col]
                .astype(str)
                .to_numpy()
            )

            classes = np.unique(
                labels
            )

            if len(classes) < 2:
                raise ValueError(
                    "Dataset harus memiliki "
                    "minimal 2 kelas unsur."
                )

            class_to_int = {
                cls: i
                for i, cls in enumerate(
                    classes
                )
            }

            y = np.array(
                [
                    class_to_int[label]
                    for label in labels
                ],
                dtype=int
            )

            if any(
                np.sum(y == cls) < 2
                for cls in range(len(classes))
            ):
                raise ValueError(
                    "Setiap kelas minimal "
                    "memerlukan 2 data."
                )

            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = stratified_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
            )

            model = SimpleRandomForest(
                n_estimators=80,
                max_depth=7,
                min_samples_split=3,
                random_state=42,
            )

            self.status_var.set(
                "Status: TRAINING RANDOM FOREST..."
            )

            self.root.update_idletasks()

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            (
                accuracy,
                precision,
                recall,
                f1,
                cm,
            ) = classification_metrics(
                y_test,
                predictions,
                len(classes),
            )

            self.model = model
            self.model_classes = list(
                classes
            )

            self.model_var.set(
                "Model siap digunakan"
            )
            if hasattr(self, "auto_analysis_var"):
                self.auto_analysis_var.set(
                    "MODEL SIAP: identifikasi otomatis aktif saat akuisisi berjalan."
                )
                self.auto_prediction_var.set(
                    "Hasil identifikasi: menunggu spektrum live."
                )

            self._show_ml_metrics(
                accuracy,
                precision,
                recall,
                f1,
                len(clean),
                len(classes),
                classes,
                cm,
            )

            self.status_var.set(
                "Status: TRAINING SELESAI"
            )

            cm_text = "\n".join(
                " | ".join(
                    str(int(v))
                    for v in row
                )
                for row in cm
            )

            messagebox.showinfo(
                "Training Berhasil",
                "Pure Python Random Forest "
                "berhasil dilatih.\n\n"
                f"Jumlah data: {len(clean)}\n"
                f"Kelas: {', '.join(classes)}\n\n"
                f"Accuracy : {accuracy:.2%}\n"
                f"Precision: {precision:.2%}\n"
                f"Recall   : {recall:.2%}\n"
                f"F1-Score : {f1:.2%}\n\n"
                "Confusion Matrix:\n"
                f"{cm_text}\n\n"
                "Metrik ini berasal dari dataset "
                "yang dipilih, bukan hasil pengukuran "
                "alat secara langsung."
            )

        except Exception as e:
            messagebox.showerror(
                "Training Gagal",
                str(e)
            )

    def _show_ml_metrics(self, accuracy, precision, recall, f1,
                         data_count, class_count, classes, cm):
        self.ml_accuracy_var.set(f"Accuracy: {accuracy:.2%}")
        self.ml_precision_var.set(f"Precision: {precision:.2%}")
        self.ml_recall_var.set(f"Recall: {recall:.2%}")
        self.ml_f1_var.set(f"F1-Score: {f1:.2%}")
        self.ml_data_var.set(f"Data: {data_count}")
        self.ml_class_var.set(
            f"Kelas: {class_count} ({', '.join(map(str, classes))})"
        )

        for item in self.cm_tree.get_children():
            self.cm_tree.delete(item)

        self.cm_tree.insert(
            "", "end",
            text="        " + "   ".join(str(c) for c in classes)
        )

        for i, row in enumerate(cm):
            values = "   ".join(str(int(v)) for v in row)
            self.cm_tree.insert(
                "", "end",
                text=f"{classes[i]}    {values}"
            )

    def identify_element(self):
        if self.model is None:
            messagebox.showwarning(
                "Identifikasi",
                "Model belum dilatih. "
                "Upload dataset training "
                "terlebih dahulu."
            )

            return

        features = (
            self.build_feature_vector()
        )

        if features is None:
            return

        probabilities = (
            self.model.predict_proba(
                [features]
            )[0]
        )

        best_index = int(
            np.argmax(probabilities)
        )

        prediction = (
            self.model_classes[
                best_index
            ]
        )

        confidence = float(
            probabilities[
                best_index
            ]
        )

        self.last_prediction = prediction
        self.last_confidence = confidence

        self.prediction_var.set(
            f"Prediksi: {prediction} | "
            f"Confidence: {confidence:.2%}"
        )

        self._refresh_result_tab()

        self.status_var.set(
            "Status: IDENTIFIKASI SELESAI"
        )

        messagebox.showinfo(
            "Hasil Identifikasi",
            f"Unsur terprediksi: {prediction}\n"
            f"Confidence: {confidence:.2%}\n\n"
            "Catatan: hasil bergantung "
            "pada dataset training."
        )

    def reset_model(self):
        self.model = None
        self.model_classes = []

        self.model_var.set(
            "Model: belum dilatih"
        )

        self.ml_accuracy_var.set("Accuracy: -")
        self.ml_precision_var.set("Precision: -")
        self.ml_recall_var.set("Recall: -")
        self.ml_f1_var.set("F1-Score: -")
        self.ml_data_var.set("Data: -")
        self.ml_class_var.set("Kelas: -")

        for item in self.cm_tree.get_children():
            self.cm_tree.delete(item)

        self.prediction_var.set(
            "Prediksi: -"
        )

        self.status_var.set(
            "Status: MODEL DIRESET"
        )

    # ---------------- CSV ----------------

    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Pilih CSV Spektrum",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        try:
            df = pd.read_csv(
                path
            )

            lower = {
                str(c).strip().lower(): c
                for c in df.columns
            }

            channel_col = lower.get(
                "channel"
            )

            count_col = lower.get(
                "count"
            )

            if (
                channel_col is None
                or count_col is None
            ):
                raise ValueError(
                    "CSV harus memiliki "
                    "kolom 'channel' dan 'count'."
                )

            new_counts = np.zeros(
                CHANNELS,
                dtype=float
            )

            for _, row in df.iterrows():
                ch = int(
                    row[channel_col]
                )

                if 0 <= ch < CHANNELS:
                    new_counts[ch] = float(
                        row[count_col]
                    )

            self.raw_counts = (
                new_counts
            )

            self.processed_counts[:] = 0

            self.peaks = (
                self.peaks.iloc[0:0]
            )

            self._refresh_channel_table()
            self._refresh_peak_table()
            self._refresh_cacahan_table()
            self._update_peak_labels()
            self._refresh_result_tab()

            self.prediction_var.set(
                "Prediksi: -"
            )

            self.update_statistics()
            self.canvas.draw_idle()

            self.status_var.set(
                f"Status: CSV LOADED - "
                f"{Path(path).name}"
            )

        except Exception as e:
            messagebox.showerror(
                "CSV Gagal",
                str(e)
            )

    def save_graph(self):
        """Simpan grafik spektrum yang sedang tampil sebagai gambar PNG atau PDF."""
        path = filedialog.asksaveasfilename(
            title="Download / Simpan Grafik Spektrum",
            defaultextension=".png",
            filetypes=[
                ("Gambar PNG", "*.png"),
                ("Dokumen PDF", "*.pdf"),
                ("Gambar SVG", "*.svg"),
            ],
            initialfile=f"PGNAA_Spectrum_{datetime.now():%Y%m%d_%H%M%S}.png",
        )

        if not path:
            return

        try:
            extension = Path(path).suffix.lower()
            if extension not in (".png", ".pdf", ".svg"):
                path += ".png"
                extension = ".png"

            # Pastikan grafik yang disimpan mencerminkan data paling baru.
            self.canvas.draw()
            self.figure.savefig(
                path,
                format=extension.lstrip("."),
                dpi=300,
                facecolor=self.figure.get_facecolor(),
                edgecolor="none",
                bbox_inches="tight",
            )
            messagebox.showinfo(
                "Grafik Berhasil Disimpan",
                f"Grafik spektrum berhasil disimpan di:\\n{path}",
            )
        except Exception as e:
            messagebox.showerror(
                "Gagal Menyimpan Grafik",
                str(e),
            )

    def save_csv(self):
        path = filedialog.asksaveasfilename(
            title="Simpan Data Spektrum",
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv")
            ],
        )

        if not path:
            return

        try:
            with open(
                path,
                "w",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.writer(f)

                writer.writerow([
                    "channel",
                    "count"
                ])

                for (
                    ch,
                    count
                ) in enumerate(
                    self.raw_counts
                ):
                    writer.writerow([
                        ch,
                        int(count)
                    ])

            messagebox.showinfo(
                "Berhasil",
                "Data spektrum berhasil disimpan."
            )

        except Exception as e:
            messagebox.showerror(
                "Gagal Menyimpan",
                str(e)
            )

    # ---------------- UI ----------------

    def update_plot(self):
        if (
            self.simulation_mode
            and self.running
        ):
            self.generate_simulated_spectrum()

        display_counts = self._remove_isolated_spikes(self.raw_counts)
        self.line.set_ydata(display_counts)
        self.processed_line.set_ydata(self.processed_counts)
        # Grafik utama menampilkan spektrum bersih saat akuisisi aktif.
        if getattr(self, "spectrum_fill", None) is not None:
            self.spectrum_fill.remove()
        self.spectrum_fill = self.ax.fill_between(
            np.arange(CHANNELS), display_counts, step="mid",
            color="#079da5", alpha=0.92
        )
        if self.running:
            self._run_automatic_analysis()

        # Dynamic vertical scale: low-count spectra must remain visible.
        # Isolated one-channel impulses are removed from the display spectrum
        # before plotting; raw acquisition values remain preserved separately.
        # Use both displayed raw and processed values to avoid clipping either.
        visible_max = max(
            float(np.max(display_counts)) if len(display_counts) else 0.0,
            float(np.max(self.processed_counts)) if len(self.processed_counts) else 0.0,
        )
        if visible_max <= 0:
            y_upper = 10.0
        else:
            y_upper = max(10.0, visible_max * 1.25)
        self.ax.set_ylim(0, y_upper)

        self.update_statistics()
        self._update_cacahan_summary()
        self._refresh_result_tab()
        self.canvas.draw_idle()

        self.root.after(
            250,
            self.update_plot
        )

    def update_statistics(self):
        total = int(
            np.sum(
                self.raw_counts
            )
        )

        max_count = float(
            np.max(
                self.raw_counts
            )
        )

        peak_channel = (
            int(
                np.argmax(
                    self.raw_counts
                )
            )
            if max_count > 0
            else None
        )

        self.total_var.set(
            f"Total Count: {total:,}"
        )

        self.max_var.set(
            f"Max Count: {int(max_count):,}"
        )

        self.channel_var.set(
            "Peak Channel: "
            f"{peak_channel if peak_channel is not None else '-'}"
        )

    def _refresh_channel_table(self):
        for item in (
            self.table.get_children()
        ):
            self.table.delete(
                item
            )

        for ch in range(
            CHANNELS
        ):
            self.table.insert(
                "",
                "end",
                values=(
                    ch,
                    f"{int(self.raw_counts[ch]):,}"
                ),
            )

    def _refresh_peak_table(self):
        for item in (
            self.peak_table.get_children()
        ):
            self.peak_table.delete(
                item
            )

        if self.peaks.empty:
            return

        for _, row in (
            self.peaks.iterrows()
        ):
            self.peak_table.insert(
                "",
                "end",
                values=(
                    int(row["Peak"]),
                    int(row["Channel"]),
                    f"{row['Intensity']:,.2f}",
                    f"{row['FWHM']:.2f}",
                    f"{row['Area']:,.2f}",
                ),
            )

    def _update_peak_labels(self):
        self.peaks_var.set(
            f"Detected Peaks: "
            f"{len(self.peaks)}"
        )

    def on_close(self):
        self.disconnect()
        self.root.destroy()


def main():
    root = tk.Tk()

    try:
        ttk.Style().theme_use(
            "vista"
        )
    except tk.TclError:
        pass

    GammaAnalyzerApp(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()
