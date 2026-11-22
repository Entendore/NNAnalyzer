#!/usr/bin/env python3
"""
Neural Network Model Analyzer v2.0
Analyzes LLMs, SLMs, and any neural network models
Supports: GGUF, Safetensors, ONNX, PyTorch, HDF5, and more
"""

import sys
import os
import json
import struct
import re
import csv
import io
import pathlib
import math
import hashlib
import time
from collections import defaultdict, OrderedDict
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple, Set

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QPushButton, QLabel, QFileDialog, QTreeWidget,
    QTreeWidgetItem, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QGroupBox, QProgressBar, QComboBox, QTextEdit,
    QFrame, QMessageBox, QLineEdit, QListWidget, QListWidgetItem,
    QAbstractItemView, QScrollArea, QSizePolicy, QTabWidget,
    QCheckBox, QSpinBox, QDoubleSpinBox, QSlider, QToolButton,
    QMenu, QMenuBar, QStatusBar, QToolTip, QDialog, QDialogButtonBox,
    QFormLayout, QProgressDialog, QShortcutButton
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QSize, QTimer, QSettings, QMimeData, QPoint
)
from PySide6.QtGui import (
    QIcon, QPainter, QColor, QFont, QBrush, QPen, QPalette,
    QAction, QDragEnterEvent, QDropEvent, QKeySequence, QShortcut,
    QTextCursor, QPixmap, QRadialGradient, QLinearGradient, QFontMetrics
)

# ── Optional dependency checks ─────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from gguf import GGUFReader
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PySide6.QtCharts import (
        QChartView, QChart, QBarSeries, QBarSet,
        QPieSeries, QValueAxis, QBarCategoryAxis, QLineSeries,
        QAreaSeries, QCategoryAxis, QLogValueAxis
    )
    HAS_CHARTS = True
except ImportError:
    HAS_CHARTS = False


# ============================================================================
# Constants & Data Models
# ============================================================================

DTYPE_SIZES = {
    'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2,
    'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1,
    'U64': 8, 'U32': 4, 'U16': 2, 'U8': 1,
    'BOOL': 0.125,
    'Q8_0': 1, 'Q4_0': 0.5625, 'Q4_1': 0.625,
    'Q5_0': 0.6875, 'Q5_1': 0.75, 'Q8_1': 1.125,
    'Q2_K': 0.3125, 'Q3_K_S': 0.4375, 'Q3_K_M': 0.5,
    'Q3_K_L': 0.5625, 'Q4_K_S': 0.625, 'Q4_K_M': 0.6875,
    'Q5_K_S': 0.75, 'Q5_K_M': 0.8125, 'Q6_K': 0.875,
    'Q8_K': 1.0625, 'IQ2_XXS': 0.25, 'IQ2_XS': 0.3125,
    'IQ3_XXS': 0.375, 'IQ1_S': 0.1875, 'IQ4_NL': 0.5625,
    'IQ4_XS': 0.5, 'IQ3_S': 0.4375, 'IQ2_S': 0.3125,
    'float64': 8, 'float32': 4, 'float16': 2, 'bfloat16': 2,
    'int64': 8, 'int32': 4, 'int16': 2, 'int8': 1,
    'uint64': 8, 'uint32': 4, 'uint16': 2, 'uint8': 1,
    'bool': 0.125,
}

# Bits per parameter for quantization estimator
DTYPE_BITS = {k: v * 8 for k, v in DTYPE_SIZES.items()}

GGUF_DTYPE_MAP = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1',
    6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1',
    10: 'Q2_K', 11: 'Q3_K_S', 12: 'Q3_K_M', 13: 'Q3_K_L',
    14: 'Q4_K_S', 15: 'Q4_K_M', 16: 'Q5_K_S', 17: 'Q5_K_M',
    18: 'Q6_K', 19: 'Q8_K', 20: 'IQ2_XXS', 21: 'IQ2_XS',
    22: 'IQ3_XXS', 23: 'IQ1_S', 24: 'IQ4_NL', 25: 'IQ3_S',
    26: 'IQ2_S', 27: 'IQ4_XS',
}

SAFETENSORS_DTYPE_MAP = {
    'BOOL': 'BOOL', 'U8': 'U8', 'I8': 'I8', 'U16': 'U16',
    'I16': 'I16', 'U32': 'U32', 'I32': 'I32', 'U64': 'U64',
    'I64': 'I64', 'F16': 'F16', 'BF16': 'BF16', 'F32': 'F32',
    'F64': 'F64',
}

ARCH_NAMES = {
    'llama': 'LLaMA', 'mistral': 'Mistral', 'qwen2': 'Qwen2',
    'qwen': 'Qwen', 'phi3': 'Phi-3', 'phi2': 'Phi-2',
    'gemma': 'Gemma', 'gemma2': 'Gemma 2', 'falcon': 'Falcon',
    'mpt': 'MPT', 'starcoder': 'StarCoder', 'codellama': 'Code LLaMA',
    'mixtral': 'Mixtral', 'deepseek': 'DeepSeek', 'yi': 'Yi',
    'claude': 'Claude', 'gpt2': 'GPT-2', 'gptj': 'GPT-J',
    'gptneox': 'GPT-NeoX', 'bert': 'BERT', 'roberta': 'RoBERTa',
    'mamba': 'Mamba', 'jamba': 'Jamba', 'command-r': 'Command-R',
    'stable-diffusion': 'Stable Diffusion', 'vit': 'ViT',
    'chatglm': 'ChatGLM', 'internlm': 'InternLM', 'baichuan': 'Baichuan',
    'solar': 'SOLAR', 'persimmon': 'Persimmon', 'plamo': 'PLaMo',
    'starcoder2': 'StarCoder2', 'xverse': 'XVerse', 'granite': 'Granite',
}

# Known GPU VRAM sizes for hardware compatibility
GPU_VRAM = OrderedDict([
    ("RTX 3060", 12), ("RTX 4060 Ti 16GB", 16),
    ("RTX 3070", 8), ("RTX 4070", 12), ("RTX 4070 Ti Super", 16),
    ("RTX 3080", 10), ("RTX 3080 12GB", 12), ("RTX 4080", 16),
    ("RTX 3090", 24), ("RTX 4090", 24),
    ("A100 40GB", 40), ("A100 80GB", 80), ("H100", 80),
    ("L4", 24), ("A10G", 24), ("T4", 16),
    ("Mac M1 8GB", 8), ("Mac M1 Pro 16GB", 16), ("Mac M1 Max 32GB", 32),
    ("Mac M2 8GB", 8), ("Mac M2 Pro 16GB", 16), ("Mac M2 Ultra 64GB", 64),
    ("Mac M3 8GB", 8), ("Mac M3 Pro 18GB", 18), ("Mac M3 Max 64GB", 64),
    ("Mac M4 16GB", 16), ("Mac M4 Pro 24GB", 24), ("Mac M4 Max 64GB", 64),
])

# Quantization target options
QUANT_OPTIONS = OrderedDict([
    ("FP32 (4 bytes/param)", 4.0),
    ("FP16/BF16 (2 bytes/param)", 2.0),
    ("Q8_0 (1.0 bytes/param)", 1.0),
    ("Q6_K (0.875 bytes/param)", 0.875),
    ("Q5_K_M (0.8125 bytes/param)", 0.8125),
    ("Q5_K_S (0.75 bytes/param)", 0.75),
    ("Q5_1 (0.75 bytes/param)", 0.75),
    ("Q4_K_M (0.6875 bytes/param)", 0.6875),
    ("Q4_K_S (0.625 bytes/param)", 0.625),
    ("Q4_1 (0.625 bytes/param)", 0.625),
    ("Q4_0 (0.5625 bytes/param)", 0.5625),
    ("Q3_K_M (0.5 bytes/param)", 0.5),
    ("IQ4_NL (0.5625 bytes/param)", 0.5625),
    ("Q3_K_S (0.4375 bytes/param)", 0.4375),
    ("Q2_K (0.3125 bytes/param)", 0.3125),
    ("IQ2_XXS (0.25 bytes/param)", 0.25),
    ("IQ1_S (0.1875 bytes/param)", 0.1875),
])

# KV cache overhead per token per layer (approximate, in bytes)
# head_dim * 2 (K+V) * 2 bytes (fp16)
KV_CACHE_BYTES_PER_TOKEN_PER_LAYER = 256  # rough average


class TensorInfo:
    """Holds metadata about a single tensor inside a model."""
    __slots__ = ('name', 'shape', 'dtype', 'data_offset', 'data_size')

    def __init__(self, name: str, shape: List[int], dtype: str,
                 data_offset: int = 0, data_size: int = 0):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.data_offset = data_offset
        self.data_size = data_size

    @property
    def num_params(self) -> int:
        p = 1
        for d in self.shape:
            p *= d
        return p

    @property
    def size_bytes(self) -> int:
        return int(self.num_params * DTYPE_SIZES.get(self.dtype, 4))

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'shape': self.shape,
            'dtype': self.dtype,
            'num_params': self.num_params,
            'size_bytes': self.size_bytes,
            'size_mb': round(self.size_mb, 4),
        }


class ModelAnalysis:
    """Aggregated analysis result for a single model file."""
    def __init__(self):
        self.file_path: str = ""
        self.file_size: int = 0
        self.format: str = ""
        self.format_version: str = ""
        self.model_name: str = ""
        self.architecture: str = ""
        self.tensors: List[TensorInfo] = []
        self.metadata: Dict[str, Any] = {}
        self.architecture_info: Dict[str, Any] = {}
        self.analysis_time: str = datetime.now().isoformat()
        self.analysis_duration_ms: float = 0
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.file_hash_md5: str = ""
        self.file_hash_sha256: str = ""

    @property
    def total_params(self) -> int:
        return sum(t.num_params for t in self.tensors)

    @property
    def total_size_bytes(self) -> int:
        return sum(t.size_bytes for t in self.tensors)

    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)

    @property
    def total_size_gb(self) -> float:
        return self.total_size_bytes / (1024 ** 3)

    @property
    def file_size_mb(self) -> float:
        return self.file_size / (1024 * 1024)

    @property
    def file_size_gb(self) -> float:
        return self.file_size / (1024 ** 3)

    @property
    def compression_ratio(self) -> float:
        """Ratio of file size to estimated uncompressed size."""
        if self.total_size_bytes == 0:
            return 0
        return self.file_size / self.total_size_bytes

    @property
    def dtype_distribution(self) -> Dict[str, int]:
        d: Dict[str, int] = defaultdict(int)
        for t in self.tensors:
            d[t.dtype] += t.num_params
        return dict(d)

    @property
    def dtype_size_distribution(self) -> Dict[str, int]:
        d: Dict[str, int] = defaultdict(int)
        for t in self.tensors:
            d[t.dtype] += t.size_bytes
        return dict(d)

    @property
    def dtype_count_distribution(self) -> Dict[str, int]:
        """Number of tensors per dtype."""
        d: Dict[str, int] = defaultdict(int)
        for t in self.tensors:
            d[t.dtype] += 1
        return dict(d)

    @property
    def layer_groups(self) -> Dict[str, List[TensorInfo]]:
        groups: Dict[str, List[TensorInfo]] = defaultdict(list)
        for t in self.tensors:
            parts = t.name.split('.')
            key = '.'.join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else 'unknown')
            groups[key].append(t)
        return dict(groups)

    @property
    def layer_group_params(self) -> Dict[str, int]:
        """Total params per layer group."""
        return {n: sum(t.num_params for t in ts) for n, ts in self.layer_groups.items()}

    @property
    def context_length(self) -> Optional[int]:
        meta = {k.lower(): v for k, v in self.metadata.items()}
        for key in ('context_length', 'max_position_embeddings', 'n_ctx',
                     'max_sequence_length', 'seq_length', 'model_max_length'):
            if key in meta:
                try:
                    return int(meta[key])
                except (ValueError, TypeError):
                    pass
        return None

    @property
    def vocab_size(self) -> Optional[int]:
        meta = {k.lower(): v for k, v in self.metadata.items()}
        for key in ('vocab_size', 'n_vocab'):
            if key in meta:
                try:
                    return int(meta[key])
                except (ValueError, TypeError):
                    pass
        for t in self.tensors:
            low = t.name.lower()
            if any(k in low for k in ('embed_tokens', 'wte', 'token_embedding')):
                if t.shape and len(t.shape) >= 1:
                    return t.shape[0]
        return None

    @property
    def num_hidden_layers(self) -> Optional[int]:
        meta = {k.lower(): v for k, v in self.metadata.items()}
        for key in ('n_layer', 'num_hidden_layers', 'num_layers',
                     'n_layers', 'num_blocks'):
            if key in meta:
                try:
                    return int(meta[key])
                except (ValueError, TypeError):
                    pass
        max_layer = -1
        for t in self.tensors:
            m = re.search(r'(?:layers?|blocks?|h)\.(\d+)\.', t.name, re.IGNORECASE)
            if m:
                max_layer = max(max_layer, int(m.group(1)))
        return max_layer + 1 if max_layer >= 0 else None

    @property
    def num_attention_heads(self) -> Optional[int]:
        meta = {k.lower(): v for k, v in self.metadata.items()}
        for key in ('num_attention_heads', 'n_head', 'n_heads'):
            if key in meta:
                try:
                    return int(meta[key])
                except (ValueError, TypeError):
                    pass
        return None

    @property
    def hidden_size(self) -> Optional[int]:
        meta = {k.lower(): v for k, v in self.metadata.items()}
        for key in ('hidden_size', 'd_model', 'n_embd'):
            if key in meta:
                try:
                    return int(meta[key])
                except (ValueError, TypeError):
                    pass
        return None

    @property
    def num_kv_heads(self) -> Optional[int]:
        meta = {k.lower(): v for k, v in self.metadata.items()}
        for key in ('num_key_value_heads', 'n_kv_heads', 'num_kv_heads'):
            if key in meta:
                try:
                    return int(meta[key])
                except (ValueError, TypeError):
                    pass
        return None

    @property
    def intermediate_size(self) -> Optional[int]:
        meta = {k.lower(): v for k, v in self.metadata.items()}
        for key in ('intermediate_size', 'n_inner', 'ffn_hidden_size'):
            if key in meta:
                try:
                    return int(meta[key])
                except (ValueError, TypeError):
                    pass
        return None

    @property
    def is_quantized(self) -> bool:
        """Check if any tensors are quantized."""
        return any(t.dtype.startswith('Q') or t.dtype.startswith('IQ') for t in self.tensors)

    @property
    def quantization_summary(self) -> str:
        """Human-readable quantization status."""
        dtypes = set(t.dtype for t in self.tensors)
        quant = [d for d in dtypes if d.startswith('Q') or d.startswith('IQ')]
        if not quant:
            return "Full Precision"
        return ", ".join(sorted(quant))

    @property
    def duplicate_tensors(self) -> List[Tuple[str, int]]:
        """Find tensor names that appear more than once."""
        counts: Dict[str, int] = defaultdict(int)
        for t in self.tensors:
            counts[t.name] += 1
        return [(n, c) for n, c in counts.items() if c > 1]

    def estimate_kv_cache(self, context_length: int) -> int:
        """Estimate KV cache size in bytes for a given context length."""
        n_layers = self.num_hidden_layers or 0
        n_heads = self.num_attention_heads or 32
        n_kv_heads = self.num_kv_heads or n_heads
        h_size = self.hidden_size or 4096
        head_dim = h_size // n_heads if n_heads > 0 else 128

        # KV cache: 2 (K+V) * n_kv_heads * head_dim * n_layers * context_length * 2 (fp16)
        return 2 * n_kv_heads * head_dim * n_layers * context_length * 2

    def estimate_vram(self, context_length: int = 4096, bytes_per_param: float = 2.0) -> Dict[str, float]:
        """Estimate VRAM requirements in GB."""
        model_size = self.total_params * bytes_per_param
        kv_cache = self.estimate_kv_cache(context_length)
        # Add ~20% overhead for CUDA/fragmentation
        overhead = (model_size + kv_cache) * 0.2

        return {
            'model_gb': model_size / (1024 ** 3),
            'kv_cache_gb': kv_cache / (1024 ** 3),
            'overhead_gb': overhead / (1024 ** 3),
            'total_gb': (model_size + kv_cache + overhead) / (1024 ** 3),
            'context_length': context_length,
        }

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def fmt_params(n: int) -> str:
        if n >= 1e12:   return f"{n/1e12:.2f}T"
        if n >= 1e9:    return f"{n/1e9:.2f}B"
        if n >= 1e6:    return f"{n/1e6:.2f}M"
        if n >= 1e3:    return f"{n/1e3:.2f}K"
        return str(n)

    @staticmethod
    def fmt_size(b: float) -> str:
        if b >= 1024**4: return f"{b/1024**4:.2f} TB"
        if b >= 1024**3: return f"{b/1024**3:.2f} GB"
        if b >= 1024**2: return f"{b/1024**2:.2f} MB"
        if b >= 1024:    return f"{b/1024:.2f} KB"
        return f"{b:.0f} B"

    @staticmethod
    def fmt_percent(value: float, total: float) -> str:
        if total == 0:
            return "0%"
        return f"{value/total*100:.1f}%"

    def infer_architecture(self) -> str:
        names = set(t.name for t in self.tensors)
        meta = {k.lower(): str(v) for k, v in self.metadata.items()}

        for meta_key in ('architecture', 'general.architecture', 'model_type'):
            if meta_key in meta:
                val = meta[meta_key].lower()
                for arch_key, arch_name in ARCH_NAMES.items():
                    if arch_key in val:
                        return arch_name
                return meta[meta_key]

        if any('q_proj' in n or 'k_proj' in n for n in names):
            if any('gate_proj' in n for n in names):
                if any('blk' in n for n in names):
                    return "Gemma-style"
                if any('v_proj' in n and 'o_proj' in n for n in names):
                    return "LLaMA / Mistral"
                return "LLaMA-like"
            if any('c_attn' in n for n in names):
                return "GPT-2 / GPT-J"
            return "GPT-style Transformer"

        if any('self_attention' in n for n in names):
            return "BERT-style"

        if any('transformer.h' in n for n in names):
            return "GPT-2 / GPT-Neo"

        if any('encoder.layer' in n for n in names):
            return "BERT / RoBERTa"

        if any('blocks' in n for n in names):
            return "Mamba-style"

        if any('ln_1' in n and 'attn' in n for n in names):
            return "GPT-2"

        if any('patch_embed' in n for n in names):
            return "Vision Transformer (ViT)"

        if any('time_embed' in n for n in names):
            return "Diffusion Model"

        if any('A' in n and 'dt_proj' in n for n in names):
            return "Mamba / SSM"

        return "Unknown"

    def compute_file_hash(self, algorithm: str = 'md5', chunk_size: int = 8192) -> str:
        """Compute file hash. Only compute once then cache."""
        attr = f'file_hash_{algorithm}'
        if getattr(self, attr, ''):
            return getattr(self, attr)
        try:
            h = hashlib.new(algorithm)
            with open(self.file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    h.update(chunk)
            val = h.hexdigest()
            setattr(self, attr, val)
            return val
        except Exception:
            return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'file_path': self.file_path,
            'file_name': os.path.basename(self.file_path),
            'file_size': self.file_size,
            'file_size_human': self.fmt_size(self.file_size),
            'format': self.format,
            'format_version': self.format_version,
            'model_name': self.model_name,
            'architecture': self.architecture,
            'total_params': self.total_params,
            'total_params_human': self.fmt_params(self.total_params),
            'total_size_bytes': self.total_size_bytes,
            'total_size_human': self.fmt_size(self.total_size_bytes),
            'compression_ratio': round(self.compression_ratio, 4),
            'is_quantized': self.is_quantized,
            'quantization': self.quantization_summary,
            'num_tensors': len(self.tensors),
            'num_layer_groups': len(self.layer_groups),
            'dtype_distribution': self.dtype_distribution,
            'dtype_size_distribution': self.dtype_size_distribution,
            'context_length': self.context_length,
            'vocab_size': self.vocab_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
            'metadata': self.metadata,
            'errors': self.errors,
            'warnings': self.warnings,
            'analysis_time': self.analysis_time,
            'analysis_duration_ms': round(self.analysis_duration_ms, 1),
            'tensors': [t.to_dict() for t in self.tensors],
        }

    def to_markdown(self) -> str:
        """Generate a markdown report."""
        lines = [
            f"# Model Analysis: {self.model_name or os.path.basename(self.file_path)}",
            "",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| File | `{self.file_path}` |",
            f"| File Size | {self.fmt_size(self.file_size)} |",
            f"| Format | {self.format} {self.format_version} |",
            f"| Architecture | {self.architecture} |",
            f"| Total Parameters | {self.fmt_params(self.total_params)} ({self.total_params:,}) |",
            f"| Estimated Uncompressed | {self.fmt_size(self.total_size_bytes)} |",
            f"| Compression Ratio | {self.compression_ratio:.2%} |",
            f"| Quantization | {self.quantization_summary} |",
            f"| Tensors | {len(self.tensors)} |",
            f"| Context Length | {self.context_length or 'Unknown'} |",
            f"| Vocab Size | {self.vocab_size or 'Unknown'} |",
            f"| Hidden Layers | {self.num_hidden_layers or 'Unknown'} |",
            f"| Hidden Size | {self.hidden_size or 'Unknown'} |",
            f"| Attention Heads | {self.num_attention_heads or 'Unknown'} |",
            f"| Analysis Time | {self.analysis_time} ({self.analysis_duration_ms:.0f}ms) |",
            "",
            "## Data Type Distribution",
            "",
            "| Dtype | Parameters | % of Total | Est. Size |",
            "|-------|-----------|------------|-----------|",
        ]
        total_p = self.total_params or 1
        for dt, cnt in sorted(self.dtype_distribution.items(), key=lambda x: -x[1]):
            sz = self.dtype_size_distribution.get(dt, 0)
            lines.append(f"| {dt} | {self.fmt_params(cnt)} | {cnt/total_p*100:.1f}% | {self.fmt_size(sz)} |")

        lines.extend(["", "## Top Layer Groups", "",
                       "| Layer | Parameters | % of Total | Tensors |",
                       "|-------|-----------|------------|---------|"])
        lg = self.layer_groups
        for n, cnt in sorted(self.layer_group_params.items(), key=lambda x: -x[1])[:30]:
            lines.append(f"| `{n}` | {self.fmt_params(cnt)} | {cnt/total_p*100:.1f}% | {len(lg[n])} |")

        if self.errors or self.warnings:
            lines.extend(["", "## Issues", ""])
            for w in self.warnings:
                lines.append(f"- ⚠️ {w}")
            for e in self.errors:
                lines.append(f"- ❌ {e}")

        return "\n".join(lines)


# ============================================================================
# Parsers
# ============================================================================

class SafetensorsParser:
    @staticmethod
    def parse(path: str) -> ModelAnalysis:
        a = ModelAnalysis()
        a.file_path = path
        a.file_size = os.path.getsize(path)
        a.format = "Safetensors"
        try:
            with open(path, 'rb') as f:
                hdr_len = struct.unpack('<Q', f.read(8))[0]
                hdr = json.loads(f.read(hdr_len).decode('utf-8'))
                for name, info in hdr.items():
                    if name == '__metadata__':
                        if isinstance(info, dict):
                            a.metadata.update(info)
                        continue
                    dt = SAFETENSORS_DTYPE_MAP.get(info.get('dtype', 'F32'), info.get('dtype', 'F32'))
                    off = info.get('data_offsets', [0, 0])
                    a.tensors.append(TensorInfo(
                        name=name, shape=info.get('shape', []),
                        dtype=dt, data_offset=8 + hdr_len + off[0],
                        data_size=off[1] - off[0]))
        except Exception as e:
            a.errors.append(f"Safetensors: {e}")
        if '__metadata__' in a.metadata and isinstance(a.metadata['__metadata__'], dict):
            meta = a.metadata['__metadata__']
            a.model_name = meta.get('name', '') or meta.get('model_name', '')
            if 'architecture' in meta:
                a.metadata['architecture'] = meta['architecture']
        a.architecture = a.infer_architecture()
        return a


class GGUFParser:
    GGUF_MAGIC = 0x46475547

    @staticmethod
    def parse(path: str) -> ModelAnalysis:
        a = ModelAnalysis()
        a.file_path = path
        a.file_size = os.path.getsize(path)
        a.format = "GGUF"
        if HAS_GGUF:
            try:
                GGUFParser._parse_with_lib(path, a)
            except Exception as e:
                a.errors.append(f"GGUF (lib): {e}")
                try:
                    a.tensors.clear(); a.metadata.clear()
                    GGUFParser._parse_manual(path, a)
                except Exception as e2:
                    a.errors.append(f"GGUF (manual): {e2}")
        else:
            try:
                GGUFParser._parse_manual(path, a)
            except Exception as e:
                a.errors.append(f"GGUF (manual): {e}")
        if 'general.name' in a.metadata:
            a.model_name = str(a.metadata['general.name'])
        a.architecture = a.infer_architecture()
        return a

    @staticmethod
    def _parse_with_lib(path, a):
        reader = GGUFReader(path)
        for tensor in reader.tensors:
            dt = GGUF_DTYPE_MAP.get(tensor.tensor_type, f"UNK_{tensor.tensor_type}")
            a.tensors.append(TensorInfo(name=tensor.name, shape=list(tensor.shape),
                                        dtype=dt, data_size=tensor.n_bytes))
        for key, field in reader.fields.items():
            try:
                val = field
                if hasattr(field, 'parts'):
                    parts = field.parts
                    if len(parts) == 1:
                        val = parts[0]
                        if hasattr(val, 'item'):
                            val = val.item()
                        elif hasattr(val, 'decode'):
                            val = val.decode('utf-8', errors='replace')
                    else:
                        val = str(field)
                a.metadata[key] = val
            except Exception:
                a.metadata[key] = "<unreadable>"

    @staticmethod
    def _parse_manual(path, a):
        with open(path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != GGUFParser.GGUF_MAGIC:
                a.errors.append("Bad GGUF magic"); return
            ver = struct.unpack('<I', f.read(4))[0]
            a.format_version = str(ver)
            n_tensor = struct.unpack('<Q', f.read(8))[0]
            n_kv = struct.unpack('<Q', f.read(8))[0]
            for _ in range(min(n_kv, 500)):
                try:
                    key = GGUFParser._rs(f)
                    vtype = struct.unpack('<I', f.read(4))[0]
                    val = GGUFParser._rv(f, vtype)
                    if key: a.metadata[key] = val
                except Exception:
                    break
            for _ in range(n_tensor):
                try:
                    name = GGUFParser._rs(f)
                    nd = struct.unpack('<I', f.read(4))[0]
                    dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
                    dt = struct.unpack('<I', f.read(4))[0]
                    off = struct.unpack('<Q', f.read(8))[0]
                    dt_s = GGUF_DTYPE_MAP.get(dt, f"UNK_{dt}")
                    a.tensors.append(TensorInfo(name=name, shape=dims, dtype=dt_s, data_offset=off))
                except Exception:
                    break

    @staticmethod
    def _rs(f) -> str:
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n).decode('utf-8', errors='replace')

    @staticmethod
    def _rv(f, vt):
        _MAP = {0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2),
                4: ('<I', 4), 5: ('<i', 4), 6: ('<f', 4), 7: ('<d', 8),
                10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8)}
        if vt == 8: return GGUFParser._rs(f)
        if vt == 9:
            at = struct.unpack('<I', f.read(4))[0]
            al = struct.unpack('<Q', f.read(8))[0]
            return [GGUFParser._rv(f, at) for _ in range(min(al, 100))]
        if vt in _MAP:
            fmt, sz = _MAP[vt]
            d = f.read(sz)
            return struct.unpack(fmt, d)[0] if len(d) == sz else None
        return None


class ONNXParser:
    @staticmethod
    def parse(path: str) -> ModelAnalysis:
        a = ModelAnalysis()
        a.file_path = path
        a.file_size = os.path.getsize(path)
        a.format = "ONNX"
        if not HAS_ONNX:
            a.errors.append("pip install onnx"); return a
        try:
            model = onnx.load(path, load_external_data=False)
            if model.opset_import:
                a.format_version = f"opset_{model.opset_import[0].version}"
            if model.producer_name:
                a.metadata['producer'] = model.producer_name
            if model.model_version:
                a.metadata['model_version'] = str(model.model_version)
            if model.doc_string:
                a.metadata['doc_string'] = model.doc_string
            DT = {1:'F32',2:'U8',3:'I8',4:'U16',5:'I16',6:'I32',7:'I64',
                  9:'BOOL',10:'F16',11:'F64',12:'U32',13:'U64'}
            for init in model.graph.initializer:
                a.tensors.append(TensorInfo(name=init.name, shape=list(init.dims),
                                            dtype=DT.get(init.data_type, f'UNK_{init.data_type}')))
            a.metadata['inputs'] = [i.name for i in model.graph.input]
            a.metadata['outputs'] = [o.name for o in model.graph.output]
            ops: Dict[str, int] = defaultdict(int)
            for node in model.graph.node:
                ops[node.op_type] += 1
            a.metadata['op_counts'] = dict(ops)
            a.metadata['num_nodes'] = len(model.graph.node)
            # Extract model metadata props
            for prop in model.metadata_props:
                a.metadata[f'meta.{prop.key}'] = prop.value
        except Exception as e:
            a.errors.append(f"ONNX: {e}")
        a.architecture = a.infer_architecture()
        return a


class PyTorchParser:
    DT_MAP = {
        'torch.float32':'F32','torch.float16':'F16','torch.float64':'F64',
        'torch.int32':'I32','torch.int64':'I64','torch.int16':'I16',
        'torch.int8':'I8','torch.uint8':'U8','torch.bfloat16':'BF16',
        'torch.bool':'BOOL','torch.float8_e4m3fn':'F8_E4M3',
        'torch.float8_e5m2':'F8_E5M2',
    }

    @staticmethod
    def parse(path: str) -> ModelAnalysis:
        a = ModelAnalysis()
        a.file_path = path
        a.file_size = os.path.getsize(path)
        a.format = "PyTorch"
        if not HAS_TORCH:
            a.errors.append("pip install torch"); return a
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            PyTorchParser._walk(ckpt, '', a)
        except Exception as e:
            a.errors.append(f"PyTorch: {e}")
            # Try safetensors fallback for .bin files
            try:
                with open(path, 'rb') as f:
                    hdr_len_bytes = f.read(8)
                    hdr_len = struct.unpack('<Q', hdr_len_bytes)[0]
                    if 0 < hdr_len < 10_000_000:
                        hdr_data = f.read(hdr_len).decode('utf-8')
                        json.loads(hdr_data)
                        a.tensors.clear(); a.metadata.clear(); a.errors.clear()
                        a.format = "Safetensors (auto-detected)"
                        return SafetensorsParser.parse(path)
            except Exception:
                pass
        a.architecture = a.infer_architecture()
        return a

    @staticmethod
    def _walk(obj, prefix, a: ModelAnalysis, depth: int = 0):
        if depth > 20:
            return
        if isinstance(obj, torch.Tensor):
            a.tensors.append(TensorInfo(
                name=prefix.lstrip('.'), shape=list(obj.shape),
                dtype=PyTorchParser.DT_MAP.get(str(obj.dtype), str(obj.dtype))))
        elif isinstance(obj, dict):
            sd = None
            for k in ('state_dict', 'model_state_dict', 'model'):
                if k in obj and isinstance(obj[k], dict):
                    sd = obj[k]; break
            if sd is not None:
                PyTorchParser._walk(sd, prefix, a, depth + 1)
            else:
                first = next(iter(obj.values()), None)
                if isinstance(first, torch.Tensor):
                    for k, v in obj.items():
                        PyTorchParser._walk(v, f"{prefix}.{k}", a, depth + 1)
                else:
                    for k, v in obj.items():
                        if isinstance(v, (dict, torch.Tensor)):
                            PyTorchParser._walk(v, f"{prefix}.{k}", a, depth + 1)
                        elif not isinstance(v, (list, set)) and depth < 3:
                            a.metadata[k] = str(v)[:500]


class HDF5Parser:
    DT_MAP = {'float32':'F32','float64':'F64','float16':'F16','int32':'I32',
              'int64':'I64','int16':'I16','int8':'I8','uint8':'U8','bool':'BOOL',
              'bfloat16':'BF16'}

    @staticmethod
    def parse(path: str) -> ModelAnalysis:
        a = ModelAnalysis()
        a.file_path = path
        a.file_size = os.path.getsize(path)
        a.format = "HDF5"
        if not HAS_H5PY:
            a.errors.append("pip install h5py"); return a
        try:
            with h5py.File(path, 'r') as f:
                for attr_name in f.attrs:
                    try:
                        a.metadata[f'root.{attr_name}'] = str(f.attrs[attr_name])
                    except Exception:
                        pass
                HDF5Parser._visit(f, '', a)
        except Exception as e:
            a.errors.append(f"HDF5: {e}")
        a.architecture = a.infer_architecture()
        return a

    @staticmethod
    def _visit(g, prefix, a):
        for k in g:
            item = g[k]
            full = f"{prefix}.{k}" if prefix else k
            if isinstance(item, h5py.Dataset):
                a.tensors.append(TensorInfo(
                    name=full, shape=list(item.shape),
                    dtype=HDF5Parser.DT_MAP.get(str(item.dtype), str(item.dtype))))
            elif isinstance(item, h5py.Group):
                for attr_name in item.attrs:
                    try:
                        a.metadata[f'{full}.{attr_name}'] = str(item.attrs[attr_name])
                    except Exception:
                        pass
                HDF5Parser._visit(item, full, a)


class GenericParser:
    @staticmethod
    def parse(path: str) -> ModelAnalysis:
        a = ModelAnalysis()
        a.file_path = path
        a.file_size = os.path.getsize(path)
        a.format = "Unknown"
        try:
            with open(path, 'rb') as f:
                hdr = f.read(256)
                a.metadata['header_hex'] = hdr[:32].hex()
                if hdr[:4] == b'PK\x03\x04':
                    a.format = "ZIP"
                    a.warnings.append("ZIP file — may be a PyTorch or TF checkpoint")
                elif hdr[:4] == b'\x89HDF':
                    a.format = "HDF5 (install h5py)"
                elif hdr[:8] == b'\x00\x00\x00\x00\x00\x00\x00\x00':
                    a.format = "Possibly PyTorch"
                    a.warnings.append("Install torch for proper parsing")
                elif hdr[:3] == b'\x1f\x8b\x08':
                    a.format = "Gzip"
                    a.warnings.append("Compressed file — may contain a model")
                elif hdr[:4] == b'OLS\x00':
                    a.format = "GGML (legacy)"
                    a.warnings.append("Legacy GGML format — consider converting to GGUF")
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read(10000)
                if txt.strip().startswith('{'):
                    a.format = "JSON"
                    try:
                        data = json.loads(txt) if len(txt) < 100000 else {}
                        a.metadata['content_preview'] = str(data)[:2000]
                    except json.JSONDecodeError:
                        a.metadata['content_preview'] = txt[:2000]
                elif txt.strip().startswith('<?xml') or txt.strip().startswith('<'):
                    a.format = "XML"
                    a.metadata['content_preview'] = txt[:2000]
                elif 'pickle' in txt[:1000].lower():
                    a.format = "Pickle"
                    a.warnings.append("Pickle file — may contain a model state dict")
            except Exception:
                pass
        except Exception as e:
            a.errors.append(f"Generic: {e}")
        return a


class ParserRegistry:
    _MAP = {
        '.safetensors': SafetensorsParser, '.gguf': GGUFParser,
        '.onnx': ONNXParser, '.pt': PyTorchParser, '.pth': PyTorchParser,
        '.bin': PyTorchParser, '.ckpt': PyTorchParser,
        '.h5': HDF5Parser, '.hdf5': HDF5Parser, '.keras': HDF5Parser,
    }

    @classmethod
    def get(cls, path: str):
        ext = pathlib.Path(path).suffix.lower()
        if ext == '.bin':
            try:
                with open(path, 'rb') as f:
                    h = f.read(8)
                    hl = struct.unpack('<Q', h)[0]
                    if 0 < hl < 10_000_000:
                        j = f.read(hl).decode('utf-8')
                        json.loads(j)
                        return SafetensorsParser
            except Exception:
                pass
        if ext == '.bin':
            try:
                with open(path, 'rb') as f:
                    magic = f.read(4)
                    if struct.unpack('<I', magic)[0] == GGUFParser.GGUF_MAGIC:
                        return GGUFParser
            except Exception:
                pass
        return cls._MAP.get(ext, GenericParser)

    @classmethod
    def filter(cls) -> str:
        return ("Model Files (*.safetensors *.gguf *.onnx *.pt *.pth *.bin "
                "*.ckpt *.h5 *.hdf5 *.keras);;All Files (*)")

    @classmethod
    def all_extensions(cls) -> List[str]:
        return list(cls._MAP.keys())


# ============================================================================
# Worker Threads
# ============================================================================

class AnalysisWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, path: str, compute_hash: bool = False):
        super().__init__()
        self.path = path
        self._cancelled = False
        self._compute_hash = compute_hash

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.progress.emit(10, "Detecting format…")
            parser = ParserRegistry.get(self.path)
            if self._cancelled: return
            self.progress.emit(30, f"Parsing ({parser.__name__})…")
            t0 = time.time()
            a = parser.parse(self.path)
            a.analysis_duration_ms = (time.time() - t0) * 1000
            if self._cancelled: return
            self.progress.emit(70, "Analysing…")
            if not a.architecture or a.architecture == "Unknown":
                a.architecture = a.infer_architecture()
            a.analysis_time = datetime.now().isoformat()
            if self._cancelled: return
            if self._compute_hash:
                self.progress.emit(85, "Computing MD5 hash…")
                a.compute_file_hash('md5')
            self.progress.emit(100, "Done")
            self.finished.emit(a)
        except Exception as e:
            if not self._cancelled:
                self.error.emit(str(e))


class BatchAnalysisWorker(QThread):
    """Analyse multiple model files in sequence."""
    progress = Signal(int, str)  # overall percent, message
    file_finished = Signal(int, object)  # index, ModelAnalysis
    finished = Signal(list)  # list of ModelAnalysis
    error = Signal(str)

    def __init__(self, paths: List[str]):
        super().__init__()
        self.paths = paths
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        results = []
        n = len(self.paths)
        for i, path in enumerate(self.paths):
            if self._cancelled: return
            pct = int((i / n) * 100)
            self.progress.emit(pct, f"[{i+1}/{n}] {os.path.basename(path)}")
            try:
                parser = ParserRegistry.get(path)
                a = parser.parse(path)
                if not a.architecture or a.architecture == "Unknown":
                    a.architecture = a.infer_architecture()
                a.analysis_time = datetime.now().isoformat()
                results.append(a)
                self.file_finished.emit(i, a)
            except Exception as e:
                self.error.emit(f"{path}: {e}")
        self.progress.emit(100, "Done")
        self.finished.emit(results)


# ============================================================================
# Theme & Styles
# ============================================================================

class C:
    BG  = "#1e1e2e"; BG2 = "#181825"; SURF = "#313244"; OVER = "#45475a"
    TXT = "#cdd6f4"; TXT2= "#a6adc8"; MUT = "#6c7086"
    ACC = "#89b4fa"; ACC2= "#cba6f7"; GRN = "#a6e3a1"
    RED = "#f38ba8"; YEL = "#f9e2af"; ORG = "#fab387"
    PNK = "#f5c2e7"; TL  = "#94e2d5"; LAV = "#b4befe"

STYLESHEET = f"""
QMainWindow {{ background:{C.BG}; color:{C.TXT}; }}
QWidget {{ background:{C.BG}; color:{C.TXT}; font-family:'Segoe UI','Inter',sans-serif; font-size:13px; }}
QGroupBox {{ background:{C.BG2}; border:1px solid {C.OVER}; border-radius:8px;
             margin-top:14px; padding:16px; padding-top:30px; font-weight:bold; color:{C.ACC}; }}
QGroupBox::title {{ subcontrol-origin:margin; left:12px; padding:0 8px; color:{C.ACC}; }}
QPushButton {{ background:{C.SURF}; border:1px solid {C.OVER}; border-radius:6px;
               padding:8px 16px; color:{C.TXT}; font-weight:500; min-height:20px; }}
QPushButton:hover {{ background:{C.OVER}; border-color:{C.ACC}; }}
QPushButton:pressed {{ background:{C.ACC}; color:{C.BG}; }}
QPushButton:disabled {{ background:{C.BG2}; color:{C.MUT}; border-color:{C.BG2}; }}
QPushButton#primary {{ background:{C.ACC}; color:{C.BG}; border:none; font-weight:bold; }}
QPushButton#primary:hover {{ background:#a6c8ff; }}
QPushButton#danger {{ background:{C.RED}; color:{C.BG}; border:none; font-weight:bold; }}
QPushButton#danger:hover {{ background:#f5a0b5; }}
QPushButton#success {{ background:{C.GRN}; color:{C.BG}; border:none; font-weight:bold; }}
QPushButton#success:hover {{ background:#bcefb8; }}
QPushButton#nav {{ background:transparent; border:none; text-align:left;
                   padding:10px 16px; border-radius:8px; font-size:13px; }}
QPushButton#nav:hover {{ background:{C.SURF}; }}
QPushButton#nav[active="true"] {{ background:{C.SURF}; color:{C.ACC};
                                   border-left:3px solid {C.ACC}; }}
QTreeWidget {{ background:{C.BG2}; border:1px solid {C.OVER}; border-radius:6px;
               color:{C.TXT}; outline:none; padding:4px; }}
QTreeWidget::item {{ padding:4px; border-radius:4px; }}
QTreeWidget::item:hover {{ background:{C.SURF}; }}
QTreeWidget::item:selected {{ background:{C.SURF}; color:{C.ACC}; }}
QHeaderView::section {{ background:{C.SURF}; color:{C.TXT2}; border:none;
                        padding:8px; font-weight:bold; font-size:11px; }}
QTableWidget {{ background:{C.BG2}; border:1px solid {C.OVER}; border-radius:6px;
                gridline-color:{C.OVER}; color:{C.TXT}; }}
QTableWidget::item {{ padding:6px; }}
QTableWidget::item:selected {{ background:{C.SURF}; color:{C.ACC}; }}
QScrollBar:vertical {{ background:{C.BG2}; width:10px; border-radius:5px; }}
QScrollBar::handle:vertical {{ background:{C.OVER}; border-radius:5px; min-height:20px; }}
QScrollBar::handle:vertical:hover {{ background:{C.MUT}; }}
QScrollBar:horizontal {{ background:{C.BG2}; height:10px; border-radius:5px; }}
QScrollBar::handle:horizontal {{ background:{C.OVER}; border-radius:5px; min-width:20px; }}
QScrollBar::add-line,QScrollBar::sub-line {{ height:0;width:0; }}
QScrollBar::add-page,QScrollBar::sub-page {{ background:none; }}
QTabWidget::pane {{ background:{C.BG2}; border:1px solid {C.OVER}; border-radius:8px; padding:8px; }}
QTabBar::tab {{ background:transparent; color:{C.MUT}; padding:8px 16px;
                border-radius:6px; margin:2px; }}
QTabBar::tab:selected {{ background:{C.SURF}; color:{C.ACC}; }}
QTabBar::tab:hover {{ color:{C.TXT}; }}
QTextEdit {{ background:{C.BG2}; border:1px solid {C.OVER}; border-radius:6px;
             color:{C.TXT}; padding:8px; font-family:'Cascadia Code','Fira Code',monospace; font-size:12px; }}
QLineEdit {{ background:{C.BG2}; border:1px solid {C.OVER}; border-radius:6px;
              padding:8px 12px; color:{C.TXT}; }}
QLineEdit:focus {{ border-color:{C.ACC}; }}
QProgressBar {{ background:{C.SURF}; border-radius:4px; text-align:center; color:{C.TXT}; height:8px; }}
QProgressBar::chunk {{ background:{C.ACC}; border-radius:4px; }}
QComboBox {{ background:{C.SURF}; border:1px solid {C.OVER}; border-radius:6px;
              padding:6px 12px; color:{C.TXT}; }}
QComboBox::drop-down {{ border:none; }}
QComboBox QAbstractItemView {{ background:{C.BG2}; color:{C.TXT}; selection-background-color:{C.SURF}; }}
QSplitter::handle {{ background:{C.OVER}; width:1px; }}
QMenu {{ background:{C.BG2}; border:1px solid {C.OVER}; border-radius:8px; padding:4px; }}
QMenu::item {{ padding:8px 24px; border-radius:4px; }}
QMenu::item:selected {{ background:{C.SURF}; }}
QStatusBar {{ background:{C.BG2}; color:{C.MUT}; border-top:1px solid {C.OVER}; padding:4px; }}
QMenuBar {{ background:{C.BG2}; color:{C.TXT}; border-bottom:1px solid {C.OVER}; }}
QMenuBar::item:selected {{ background:{C.SURF}; border-radius:4px; }}
QListWidget {{ background:transparent; border:none; color:{C.TXT2}; font-size:11px; }}
QListWidget::item {{ padding:6px; border-radius:4px; }}
QListWidget::item:hover {{ background:{C.SURF}; }}
QToolTip {{ background:{C.BG2}; color:{C.TXT}; border:1px solid {C.OVER}; border-radius:4px; padding:4px; }}
QCheckBox {{ spacing:8px; color:{C.TXT}; }}
QCheckBox::indicator {{ width:16px; height:16px; border-radius:4px;
                        border:2px solid {C.OVER}; background:{C.BG2}; }}
QCheckBox::indicator:checked {{ background:{C.ACC}; border-color:{C.ACC}; }}
QSpinBox {{ background:{C.BG2}; border:1px solid {C.OVER}; border-radius:6px;
             padding:6px; color:{C.TXT}; }}
QSlider::groove:horizontal {{ background:{C.SURF}; height:6px; border-radius:3px; }}
QSlider::handle:horizontal {{ background:{C.ACC}; width:16px; height:16px;
                               margin:-5px 0; border-radius:8px; }}
QSlider::handle:horizontal:hover {{ background:#a6c8ff; }}
QDialog {{ background:{C.BG}; }}
QLabel#sectionLabel {{ font-size:11px; color:{C.MUT}; letter-spacing:1px;
                       text-transform:uppercase; background:transparent; }}
"""


# ============================================================================
# Reusable Widgets
# ============================================================================

class NumericItem(QTableWidgetItem):
    def __init__(self, text: str, value: float):
        super().__init__(text)
        self._value = value

    def __lt__(self, other):
        if isinstance(other, NumericItem):
            return self._value < other._value
        return super().__lt__(other)


class StatCard(QFrame):
    def __init__(self, label: str, value: str = "—", color: str = C.ACC,
                 subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"StatCard, QFrame {{ background:{C.BG2}; border:1px solid {C.OVER}; border-radius:12px; }}"
        )
        ly = QVBoxLayout(self); ly.setContentsMargins(16,16,16,16); ly.setSpacing(4)
        self._val = QLabel(value)
        self._val.setStyleSheet(f"font-size:24px;font-weight:bold;color:{color};background:transparent;")
        self._val.setWordWrap(True)
        nm = QLabel(label.upper())
        nm.setObjectName("sectionLabel")
        nm.setStyleSheet(f"font-size:10px;color:{C.MUT};letter-spacing:1px;background:transparent;")
        self._sub = QLabel(subtitle)
        self._sub.setStyleSheet(f"font-size:11px;color:{C.MUT};background:transparent;")
        ly.addWidget(self._val); ly.addWidget(nm)
        if subtitle:
            ly.addWidget(self._sub)
        ly.addStretch()

    def set_value(self, v: str, subtitle: str = ""):
        self._val.setText(v)
        if subtitle:
            self._sub.setText(subtitle)


class MiniBar(QFrame):
    """A tiny inline progress bar for showing proportions."""
    def __init__(self, value: float, max_val: float, color: str = C.ACC,
                 width: int = 120, height: int = 8, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self._value = value
        self._max = max_val
        self._color = color

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # Background
        p.setBrush(QBrush(QColor(C.SURF)))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect(), 4, 4)
        # Fill
        if self._max > 0:
            ratio = min(self._value / self._max, 1.0)
            w = int(self.width() * ratio)
            p.setBrush(QBrush(QColor(self._color)))
            p.drawRoundedRect(0, 0, w, self.height(), 4, 4)
        p.end()


class ClickableLabel(QLabel):
    """A label that emits a signal when clicked."""
    clicked = Signal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


# ============================================================================
# Dialogs
# ============================================================================

class ExportDialog(QDialog):
    """Dialog for exporting analysis results."""
    def __init__(self, analysis: ModelAnalysis, parent=None):
        super().__init__(parent)
        self.analysis = analysis
        self.setWindowTitle("Export Analysis")
        self.setMinimumWidth(450)
        self.setStyleSheet(f"QDialog {{ background:{C.BG}; }}")

        ly = QVBoxLayout(self); ly.setSpacing(16)

        title = QLabel("Export Analysis Results")
        title.setStyleSheet(f"font-size:18px;font-weight:bold;color:{C.TXT};background:transparent;")
        ly.addWidget(title)

        # Format selection
        form = QFormLayout(); form.setSpacing(12)
        self.format_cb = QComboBox()
        self.format_cb.addItems(["JSON", "Markdown", "CSV (tensors)", "CSV (full)"])
        form.addRow("Format:", self.format_cb)

        self.include_tensors = QCheckBox("Include tensor details")
        self.include_tensors.setChecked(True)
        form.addRow(self.include_tensors)

        self.include_metadata = QCheckBox("Include metadata")
        self.include_metadata.setChecked(True)
        form.addRow(self.include_metadata)

        self.include_hashes = QCheckBox("Include file hashes (slow)")
        self.include_hashes.setChecked(False)
        form.addRow(self.include_hashes)

        ly.addLayout(form)

        # Preview
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(200)
        self.preview.setStyleSheet(f"font-size:11px;")
        ly.addWidget(QLabel("Preview:"))
        ly.addWidget(self.preview)

        self.format_cb.currentTextChanged.connect(self._update_preview)
        self._update_preview()

        # Buttons
        btns = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        export_btn = QPushButton("Export…")
        export_btn.setObjectName("primary")
        export_btn.clicked.connect(self._do_export)
        btns.addStretch(); btns.addWidget(cancel_btn); btns.addWidget(export_btn)
        ly.addLayout(btns)

    def _get_content(self) -> str:
        fmt = self.format_cb.currentText()
        a = self.analysis

        if fmt == "JSON":
            d = a.to_dict()
            if not self.include_tensors.isChecked():
                d.pop('tensors', None)
            if not self.include_metadata.isChecked():
                d.pop('metadata', None)
            if self.include_hashes.isChecked():
                d['md5'] = a.compute_file_hash('md5')
            return json.dumps(d, indent=2, default=str)

        elif fmt == "Markdown":
            return a.to_markdown()

        elif fmt.startswith("CSV"):
            output = io.StringIO()
            writer = csv.writer(output)
            if "full" in fmt:
                # Full CSV with model info + tensors
                writer.writerow(["# Model Analysis"])
                writer.writerow(["Property", "Value"])
                writer.writerow(["File", a.file_path])
                writer.writerow(["Format", a.format])
                writer.writerow(["Architecture", a.architecture])
                writer.writerow(["Total Params", a.total_params])
                writer.writerow(["File Size", a.fmt_size(a.file_size)])
                writer.writerow([])
                writer.writerow(["# Tensors"])
            writer.writerow(["Name", "Shape", "Dtype", "Parameters", "Size (bytes)", "Size (MB)"])
            for t in a.tensors:
                writer.writerow([t.name, t.shape, t.dtype, t.num_params, t.size_bytes, f"{t.size_mb:.4f}"])
            return output.getvalue()

        return ""

    def _update_preview(self):
        content = self._get_content()
        self.preview.setPlainText(content[:3000] + ("…" if len(content) > 3000 else ""))

    def _do_export(self):
        fmt = self.format_cb.currentText()
        ext_map = {"JSON": ".json", "Markdown": ".md", "CSV (tensors)": ".csv", "CSV (full)": ".csv"}
        default_name = os.path.splitext(os.path.basename(self.analysis.file_path))[0] + ext_map.get(fmt, ".txt")
        filter_map = {"JSON": "JSON (*.json)", "Markdown": "Markdown (*.md)",
                      "CSV (tensors)": "CSV (*.csv)", "CSV (full)": "CSV (*.csv)"}
        path, _ = QFileDialog.getSaveFileName(self, "Export", default_name, filter_map.get(fmt, ""))
        if path:
            try:
                content = self._get_content()
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                QMessageBox.information(self, "Exported", f"Saved to {path}")
                self.accept()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))


class BatchDialog(QDialog):
    """Dialog for batch analysis of multiple files."""
    files_selected = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Analysis")
        self.setMinimumSize(550, 400)
        self.setStyleSheet(f"QDialog {{ background:{C.BG}; }}")

        ly = QVBoxLayout(self); ly.setSpacing(12)

        title = QLabel("Batch Analysis")
        title.setStyleSheet(f"font-size:18px;font-weight:bold;color:{C.TXT};background:transparent;")
        ly.addWidget(title)

        desc = QLabel("Select multiple model files to analyse at once.")
        desc.setStyleSheet(f"color:{C.TXT2};background:transparent;")
        ly.addWidget(desc)

        # File list
        self.file_list = QListWidget()
        self.file_list.setStyleSheet(
            f"QListWidget {{ background:{C.BG2}; border:1px solid {C.OVER}; border-radius:6px; "
            f"color:{C.TXT}; padding:4px; }}"
        )
        ly.addWidget(self.file_list)

        # Buttons
        btns = QHBoxLayout()
        add_btn = QPushButton("Add Files…")
        add_btn.clicked.connect(self._add_files)
        rm_btn = QPushButton("Remove Selected")
        rm_btn.clicked.connect(self._remove_selected)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.file_list.clear)
        btns.addWidget(add_btn); btns.addWidget(rm_btn); btns.addWidget(clear_btn)
        btns.addStretch()
        ly.addLayout(btns)

        # Start
        start_btn = QPushButton("Start Analysis")
        start_btn.setObjectName("primary")
        start_btn.clicked.connect(self._start)
        ly.addWidget(start_btn)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Model Files", "", ParserRegistry.filter())
        for p in paths:
            self.file_list.addItem(p)

    def _remove_selected(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    def _start(self):
        paths = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if paths:
            self.files_selected.emit(paths)
            self.accept()


# ============================================================================
# Page Widgets
# ============================================================================

class WelcomePage(QWidget):
    file_dropped = Signal(str)
    file_selected = Signal(str)
    batch_requested = Signal()

    def __init__(self, recent_files: List[str] = None):
        super().__init__()
        self.setAcceptDrops(True)
        ly = QVBoxLayout(self); ly.setAlignment(Qt.AlignCenter); ly.setSpacing(24)
        ly.addWidget(self._lbl("🧠", 72, True, Qt.AlignCenter))
        ly.addWidget(self._lbl("Neural Network Model Analyzer", 28, True, Qt.AlignCenter, C.TXT, True))
        ly.addWidget(self._lbl(
            "Analyse LLMs · SLMs · any neural network architecture\n"
            "GGUF · Safetensors · ONNX · PyTorch · HDF5 & more",
            14, False, Qt.AlignCenter, C.TXT2))

        # Drop zone
        self.drop = QFrame()
        self.drop.setStyleSheet(
            f"QFrame{{border:2px dashed {C.OVER};border-radius:16px;"
            f"background:{C.BG2};padding:40px;}}"
        )
        dl = QVBoxLayout(self.drop); dl.setAlignment(Qt.AlignCenter); dl.setSpacing(12)
        dl.addWidget(self._lbl("📂", 48, True, Qt.AlignCenter))
        dl.addWidget(self._lbl("Drop a model file here", 16, False, Qt.AlignCenter, C.TXT2))
        dl.addWidget(self._lbl("or", 12, False, Qt.AlignCenter, C.MUT))
        b = QPushButton("Browse Files"); b.setObjectName("primary"); b.setCursor(Qt.PointingHandCursor)
        b.setFixedWidth(160); b.clicked.connect(self._browse); dl.addWidget(b, alignment=Qt.AlignCenter)

        # Batch button
        bb = QPushButton("Batch Analysis…"); bb.setCursor(Qt.PointingHandCursor)
        bb.setFixedWidth(160); bb.clicked.connect(self.batch_requested.emit)
        dl.addWidget(bb, alignment=Qt.AlignCenter)

        ly.addWidget(self.drop, alignment=Qt.AlignCenter)

        # Recent files
        recent = recent_files or []
        if recent:
            rl = QVBoxLayout(); rl.setSpacing(8)
            rl.addWidget(self._lbl("Recent Files", 13, True, Qt.AlignCenter, C.TXT2))
            for rf in recent[:5]:
                btn = QPushButton(f"📄 {os.path.basename(rf)}")
                btn.setObjectName("nav")
                btn.setCursor(Qt.PointingHandCursor)
                btn.setToolTip(rf)
                btn.clicked.connect(lambda checked, p=rf: self.file_selected.emit(p))
                rl.addWidget(btn)
                if rl.count() > 6:
                    break
            ly.addLayout(rl)

        # Supported formats & dependencies
        fmts_text = "Supported: " + " · ".join(ParserRegistry.all_extensions())
        ly.addWidget(self._lbl(fmts_text, 11, False, Qt.AlignCenter, C.MUT))

        deps = "  ".join(f"{'✅' if h else '❌'} {n}" for n, h in
                         [("PySide6",True),("PyTorch",HAS_TORCH),("Safetensors",HAS_SAFETENSORS),
                          ("ONNX",HAS_ONNX),("HDF5",HAS_H5PY),("GGUF",HAS_GGUF),("NumPy",HAS_NUMPY)])
        ly.addWidget(self._lbl(f"Libraries: {deps}", 10, False, Qt.AlignCenter, C.MUT))

        ver = self._lbl("v2.0", 10, False, Qt.AlignCenter, C.MUT)
        ly.addWidget(ver)

    @staticmethod
    def _lbl(text, sz=13, bold=False, align=Qt.AlignLeft, color=C.TXT, is_icon=False):
        l = QLabel(text); l.setAlignment(align)
        if is_icon:
            l.setStyleSheet(f"font-size:{sz}px;background:transparent;")
        else:
            l.setStyleSheet(
                f"font-size:{sz}px;{'font-weight:bold;' if bold else ''}"
                f"color:{color};background:transparent;"
            )
        l.setWordWrap(True); return l

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Model", "", ParserRegistry.filter())
        if p: self.file_selected.emit(p)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.drop.setStyleSheet(
                f"QFrame{{border:2px dashed {C.ACC};border-radius:16px;"
                f"background:{C.SURF};padding:40px;}}"
            )

    def dragLeaveEvent(self, e):
        self.drop.setStyleSheet(
            f"QFrame{{border:2px dashed {C.OVER};border-radius:16px;"
            f"background:{C.BG2};padding:40px;}}"
        )

    def dropEvent(self, e: QDropEvent):
        self.dragLeaveEvent(e)
        urls = e.mimeData().urls()
        if urls and urls[0].toLocalFile():
            self.file_dropped.emit(urls[0].toLocalFile())


class OverviewPage(QWidget):
    def __init__(self):
        super().__init__()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        c = QWidget()
        self.ml = QVBoxLayout(c); self.ml.setSpacing(16); self.ml.setContentsMargins(8,8,8,8)

        # Header
        hf = QFrame()
        hf.setStyleSheet(f"QFrame{{background:{C.BG2};border-radius:12px;padding:24px;}}")
        hl = QHBoxLayout(hf)
        self.icon = QLabel("🧠"); self.icon.setStyleSheet("font-size:48px;background:transparent;")
        ht = QVBoxLayout()
        self.name_l = QLabel("No model loaded")
        self.name_l.setStyleSheet(f"font-size:22px;font-weight:bold;color:{C.TXT};background:transparent;")
        self.arch_l = QLabel("")
        self.arch_l.setStyleSheet(f"font-size:13px;color:{C.TXT2};background:transparent;")

        # Format + Quant badges
        badge_row = QHBoxLayout()
        self.fmt_l = QLabel("")
        self.fmt_l.setStyleSheet(
            f"background:{C.ACC};color:{C.BG};border-radius:4px;"
            f"padding:4px 10px;font-weight:bold;font-size:11px;"
        )
        self.fmt_l.setFixedHeight(24)
        self.quant_l = QLabel("")
        self.quant_l.setStyleSheet(
            f"background:{C.ORG};color:{C.BG};border-radius:4px;"
            f"padding:4px 10px;font-weight:bold;font-size:11px;"
        )
        self.quant_l.setFixedHeight(24)
        badge_row.addWidget(self.fmt_l); badge_row.addWidget(self.quant_l)
        badge_row.addStretch()

        self.path_l = QLabel("")
        self.path_l.setStyleSheet(f"font-size:11px;color:{C.MUT};background:transparent;")
        self.path_l.setWordWrap(True)
        self.time_l = QLabel("")
        self.time_l.setStyleSheet(f"font-size:11px;color:{C.MUT};background:transparent;")

        ht.addWidget(self.name_l); ht.addWidget(self.arch_l)
        ht.addLayout(badge_row); ht.addWidget(self.path_l); ht.addWidget(self.time_l)
        ht.addStretch()
        hl.addWidget(self.icon); hl.addLayout(ht, stretch=1)

        # Stat cards row 1
        sl = QHBoxLayout(); sl.setSpacing(12)
        self.cards = {}
        for key, label, col in [
            ("params", "Total Parameters", C.ACC),
            ("mem", "Est. Uncompressed", C.GRN),
            ("fsize", "File Size", C.ORG),
            ("n_tensor", "Tensors", C.ACC2),
            ("n_layer", "Layer Groups", C.TL),
        ]:
            card = StatCard(label, "—", col); self.cards[key] = card; sl.addWidget(card)

        # Stat cards row 2 — model info
        sl2 = QHBoxLayout(); sl2.setSpacing(12)
        self.cards2 = {}
        for key, label, col in [
            ("context", "Context Length", C.LAV),
            ("vocab", "Vocab Size", C.PNK),
            ("hidden_layers", "Hidden Layers", C.YEL),
            ("heads", "Attention Heads", C.TL),
            ("hidden_size", "Hidden Size", C.ACC),
        ]:
            card = StatCard(label, "—", col); self.cards2[key] = card; sl2.addWidget(card)

        # Stat cards row 3 — file info
        sl3 = QHBoxLayout(); sl3.setSpacing(12)
        self.cards3 = {}
        for key, label, col in [
            ("compression", "Compression", C.GRN),
            ("quant_status", "Quantization", C.ORG),
            ("duration", "Parse Time", C.MUT),
        ]:
            card = StatCard(label, "—", col); self.cards3[key] = card; sl3.addWidget(card)

        # Charts
        self.dtype_grp = QGroupBox("Data Type Distribution")
        self.dtype_v = self._chart_view()
        QVBoxLayout(self.dtype_grp).addWidget(self.dtype_v)

        self.size_grp = QGroupBox("Size by Data Type")
        self.size_v = self._chart_view()
        QVBoxLayout(self.size_grp).addWidget(self.size_v)

        self.layer_grp = QGroupBox("Top Layer Groups by Parameters")
        self.layer_v = self._chart_view()
        QVBoxLayout(self.layer_grp).addWidget(self.layer_v)

        cl = QHBoxLayout(); cl.addWidget(self.dtype_grp, 1); cl.addWidget(self.size_grp, 1)
        cl2 = QHBoxLayout(); cl2.addWidget(self.layer_grp, 1)

        # Errors
        self.err_grp = QGroupBox("Warnings & Errors")
        self.err_txt = QTextEdit(); self.err_txt.setReadOnly(True)
        self.err_txt.setMaximumHeight(120)
        QVBoxLayout(self.err_grp).addWidget(self.err_txt)

        self.ml.addWidget(hf)
        self.ml.addLayout(sl); self.ml.addLayout(sl2); self.ml.addLayout(sl3)
        self.ml.addLayout(cl); self.ml.addLayout(cl2)
        self.ml.addWidget(self.err_grp); self.ml.addStretch()
        scroll.setWidget(c)
        ol = QVBoxLayout(self); ol.setContentsMargins(0,0,0,0); ol.addWidget(scroll)

    def _chart_view(self):
        if HAS_CHARTS:
            v = QChartView(); v.setRenderHint(QPainter.Antialiasing)
            v.setMinimumHeight(280); v.setStyleSheet("background:transparent;")
            return v
        l = QLabel("QtCharts unavailable\npip install PySide6-addons")
        l.setAlignment(Qt.AlignCenter)
        l.setStyleSheet(f"color:{C.MUT};background:transparent;")
        return l

    def update_analysis(self, a: ModelAnalysis):
        self.cards["params"].set_value(a.fmt_params(a.total_params))
        self.cards["mem"].set_value(a.fmt_size(a.total_size_bytes))
        self.cards["fsize"].set_value(a.fmt_size(a.file_size))
        self.cards["n_tensor"].set_value(str(len(a.tensors)))
        self.cards["n_layer"].set_value(str(len(a.layer_groups)))

        ctx = a.context_length
        self.cards2["context"].set_value(f"{ctx:,}" if ctx else "—")
        vocab = a.vocab_size
        self.cards2["vocab"].set_value(f"{vocab:,}" if vocab else "—")
        layers = a.num_hidden_layers
        self.cards2["hidden_layers"].set_value(str(layers) if layers else "—")
        heads = a.num_attention_heads
        self.cards2["heads"].set_value(str(heads) if heads else "—")
        hsize = a.hidden_size
        self.cards2["hidden_size"].set_value(str(hsize) if hsize else "—")

        self.cards3["compression"].set_value(
            f"{a.compression_ratio:.1%}" if a.compression_ratio else "—",
            subtitle=a.fmt_size(a.file_size) + " / " + a.fmt_size(a.total_size_bytes)
        )
        self.cards3["quant_status"].set_value(
            "Yes" if a.is_quantized else "No",
            subtitle=a.quantization_summary
        )
        self.cards3["duration"].set_value(
            f"{a.analysis_duration_ms:.0f}ms" if a.analysis_duration_ms else "—"
        )

        self.name_l.setText(a.model_name or os.path.basename(a.file_path))
        self.arch_l.setText(f"Architecture: {a.architecture}")
        self.fmt_l.setText(f" {a.format} {a.format_version} ".strip())
        self.quant_l.setText(f" {a.quantization_summary} ")
        self.quant_l.setVisible(a.is_quantized)
        self.path_l.setText(a.file_path)
        self.time_l.setText(f"Analysed: {a.analysis_time}")

        all_msgs = []
        for w in getattr(a, 'warnings', []):
            all_msgs.append(f"⚠️  {w}")
        for e in a.errors:
            all_msgs.append(f"❌  {e}")
        if all_msgs:
            self.err_grp.setVisible(True)
            self.err_txt.setText("\n".join(all_msgs))
        else:
            self.err_grp.setVisible(False)

        if HAS_CHARTS:
            self._dtype_chart(a)
            self._size_chart(a)
            self._layer_chart(a)

    def _dtype_chart(self, a):
        ch = QChart()
        ch.setTitle("Parameters by Data Type")
        ch.setAnimationOptions(QChart.SeriesAnimations)
        ch.layout().setContentsMargins(0,0,0,0)
        ch.setBackgroundRoundness(8)
        s = QPieSeries()
        colors = [C.ACC, C.GRN, C.ORG, C.ACC2, C.TL, C.YEL, C.RED, C.PNK, C.LAV]
        for i, (dt, cnt) in enumerate(sorted(a.dtype_distribution.items(), key=lambda x: -x[1])):
            sl = s.append(f"{dt} ({a.fmt_params(cnt)})", cnt)
            sl.setBrush(QBrush(QColor(colors[i % len(colors)])))
            sl.setLabelVisible(True)
        ch.addSeries(s)
        ch.legend().setAlignment(Qt.AlignBottom)
        ch.legend().setLabelColor(QColor(C.TXT2))
        self.dtype_v.setChart(ch)

    def _size_chart(self, a):
        ch = QChart()
        ch.setTitle("Size by Data Type")
        ch.setAnimationOptions(QChart.SeriesAnimations)
        ch.layout().setContentsMargins(0,0,0,0)
        ch.setBackgroundRoundness(8)
        s = QPieSeries()
        colors = [C.ACC, C.GRN, C.ORG, C.ACC2, C.TL, C.YEL, C.RED, C.PNK, C.LAV]
        for i, (dt, sz) in enumerate(sorted(a.dtype_size_distribution.items(), key=lambda x: -x[1])):
            sl = s.append(f"{dt} ({a.fmt_size(sz)})", sz)
            sl.setBrush(QBrush(QColor(colors[(i+3) % len(colors)])))
            sl.setLabelVisible(True)
        ch.addSeries(s)
        ch.legend().setAlignment(Qt.AlignBottom)
        ch.legend().setLabelColor(QColor(C.TXT2))
        self.size_v.setChart(ch)

    def _layer_chart(self, a):
        ch = QChart()
        ch.setTitle("Top 20 Layer Groups")
        ch.setAnimationOptions(QChart.SeriesAnimations)
        ch.layout().setContentsMargins(0,0,0,0)
        ch.setBackgroundRoundness(8)
        gs = {n: sum(t.num_params for t in ts) for n, ts in a.layer_groups.items()}
        top = sorted(gs.items(), key=lambda x: -x[1])[:20]
        bs = QBarSet("Params"); cats = []
        for n, sz in top:
            bs.append(sz)
            cats.append(n if len(n) < 25 else '…' + n[-22:])
        series = QBarSeries(); series.append(bs)
        bs.setBrush(QBrush(QColor(C.ACC)))
        ch.addSeries(series)
        ax = QBarCategoryAxis(); ax.append(cats)
        ax.setLabelsAngle(-45); ax.setLabelsColor(QColor(C.TXT2))
        ch.addAxis(ax, Qt.AlignBottom); series.attachAxis(ax)
        ay = QValueAxis(); ay.setLabelsColor(QColor(C.TXT2))
        ch.addAxis(ay, Qt.AlignLeft); series.attachAxis(ay)
        ch.legend().setVisible(False)
        self.layer_v.setChart(ch)


class TensorsPage(QWidget):
    def __init__(self):
        super().__init__()
        self.analysis: Optional[ModelAnalysis] = None
        self._all_tensors: List[TensorInfo] = []
        ly = QVBoxLayout(self); ly.setSpacing(8)

        # Search bar
        sl = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search tensors (regex supported)…")
        self.search.textChanged.connect(self._filter)
        self.search.setClearButtonEnabled(True)

        self.dtype_cb = QComboBox(); self.dtype_cb.addItem("All Dtypes")
        self.dtype_cb.setMinimumWidth(120)
        self.dtype_cb.currentTextChanged.connect(self._filter)

        self.layer_cb = QComboBox(); self.layer_cb.addItem("All Layers")
        self.layer_cb.setMinimumWidth(150)
        self.layer_cb.currentTextChanged.connect(self._filter)

        self.size_filter_cb = QComboBox()
        self.size_filter_cb.addItems(["Any Size", "< 1KB", "1KB - 1MB", "1MB - 100MB", "> 100MB"])
        self.size_filter_cb.currentTextChanged.connect(self._filter)

        self.cnt_l = QLabel("0 tensors")
        self.cnt_l.setStyleSheet(f"color:{C.MUT};font-size:11px;background:transparent;")

        sl.addWidget(QLabel("🔍")); sl.addWidget(self.search, 1)
        sl.addWidget(QLabel("Dtype:")); sl.addWidget(self.dtype_cb)
        sl.addWidget(QLabel("Layer:")); sl.addWidget(self.layer_cb)
        sl.addWidget(QLabel("Size:")); sl.addWidget(self.size_filter_cb)
        sl.addWidget(self.cnt_l)

        # Export button
        self.export_btn = QPushButton("📋 Copy Table")
        self.export_btn.setFixedWidth(110)
        self.export_btn.clicked.connect(self._copy_table)
        sl.addWidget(self.export_btn)

        self.tbl = QTableWidget(); self.tbl.setColumnCount(6)
        self.tbl.setHorizontalHeaderLabels(["Name", "Shape", "Dtype", "Parameters", "Est. Size", "% of Total"])
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 6):
            self.tbl.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setSortingEnabled(True)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setStyleSheet(f"QTableWidget{{alternate-background-color:{C.BG};}}")

        # Context menu
        self.tbl.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl.customContextMenuRequested.connect(self._context_menu)

        ly.addLayout(sl); ly.addWidget(self.tbl)

    def update_analysis(self, a: ModelAnalysis):
        self.analysis = a
        self._all_tensors = a.tensors

        self.dtype_cb.blockSignals(True); self.dtype_cb.clear()
        self.dtype_cb.addItem("All Dtypes")
        for dt in sorted(a.dtype_distribution.keys()):
            self.dtype_cb.addItem(dt)
        self.dtype_cb.blockSignals(False)

        self.layer_cb.blockSignals(True); self.layer_cb.clear()
        self.layer_cb.addItem("All Layers")
        for lg in sorted(a.layer_groups.keys()):
            self.layer_cb.addItem(lg)
        self.layer_cb.blockSignals(False)

        self._populate(a.tensors)

    def _populate(self, tensors: List[TensorInfo]):
        self.tbl.setSortingEnabled(False)
        self.tbl.setRowCount(len(tensors))
        fmt = ModelAnalysis.fmt_params
        fmts = ModelAnalysis.fmt_size
        total_p = self.analysis.total_params if self.analysis else 1

        for r, t in enumerate(tensors):
            it_name = QTableWidgetItem(t.name)
            it_name.setFlags(it_name.flags() & ~Qt.ItemIsEditable)
            it_name.setToolTip(t.name)

            it_shape = QTableWidgetItem(str(t.shape))
            it_shape.setFlags(it_shape.flags() & ~Qt.ItemIsEditable)

            it_dtype = QTableWidgetItem(t.dtype)
            it_dtype.setFlags(it_dtype.flags() & ~Qt.ItemIsEditable)

            it_params = NumericItem(fmt(t.num_params), t.num_params)
            it_params.setFlags(it_params.flags() & ~Qt.ItemIsEditable)

            it_size = NumericItem(fmts(t.size_bytes), t.size_bytes)
            it_size.setFlags(it_size.flags() & ~Qt.ItemIsEditable)

            pct = t.num_params / total_p * 100 if total_p > 0 else 0
            it_pct = NumericItem(f"{pct:.2f}%", pct)
            it_pct.setFlags(it_pct.flags() & ~Qt.ItemIsEditable)

            # Color-code dtype
            if t.dtype.startswith('Q') or t.dtype.startswith('IQ'):
                it_dtype.setForeground(QBrush(QColor(C.ORG)))
                it_dtype.setToolTip("Quantized type")
            elif t.dtype in ('F16', 'BF16'):
                it_dtype.setForeground(QBrush(QColor(C.TL)))
            elif t.dtype == 'F32':
                it_dtype.setForeground(QBrush(QColor(C.GRN)))
            elif t.dtype == 'F64':
                it_dtype.setForeground(QBrush(QColor(C.ACC)))

            self.tbl.setItem(r, 0, it_name)
            self.tbl.setItem(r, 1, it_shape)
            self.tbl.setItem(r, 2, it_dtype)
            self.tbl.setItem(r, 3, it_params)
            self.tbl.setItem(r, 4, it_size)
            self.tbl.setItem(r, 5, it_pct)
        self.tbl.setSortingEnabled(True)
        self.cnt_l.setText(f"{len(tensors)} tensors")

    def _filter(self):
        if not self.analysis:
            return
        txt = self.search.text().strip()
        dt_filter = self.dtype_cb.currentText()
        layer_filter = self.layer_cb.currentText()
        size_filter = self.size_filter_cb.currentText()

        try:
            pattern = re.compile(txt, re.IGNORECASE) if txt else None
        except re.error:
            pattern = None

        filtered = []
        for t in self._all_tensors:
            if txt:
                if pattern:
                    if not pattern.search(t.name):
                        continue
                elif txt.lower() not in t.name.lower():
                    continue
            if dt_filter != "All Dtypes" and t.dtype != dt_filter:
                continue
            if layer_filter != "All Layers":
                if not t.name.startswith(layer_filter):
                    continue
            if size_filter != "Any Size":
                mb = t.size_bytes / (1024 * 1024)
                kb = t.size_bytes / 1024
                if size_filter == "< 1KB" and kb >= 1:
                    continue
                elif size_filter == "1KB - 1MB" and not (1 <= kb <= 1024):
                    continue
                elif size_filter == "1MB - 100MB" and not (1 <= mb <= 100):
                    continue
                elif size_filter == "> 100MB" and mb <= 100:
                    continue
            filtered.append(t)
        self._populate(filtered)

    def _copy_table(self):
        rows = []
        headers = []
        for col in range(self.tbl.columnCount()):
            headers.append(self.tbl.horizontalHeaderItem(col).text())
        rows.append("\t".join(headers))
        for row in range(self.tbl.rowCount()):
            cells = []
            for col in range(self.tbl.columnCount()):
                item = self.tbl.item(row, col)
                cells.append(item.text() if item else "")
            rows.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(rows))

    def _context_menu(self, pos: QPoint):
        menu = QMenu(self)
        item = self.tbl.itemAt(pos)
        if item:
            row = item.row()
            name_item = self.tbl.item(row, 0)
            if name_item:
                copy_name = menu.addAction("Copy tensor name")
                copy_name.triggered.connect(
                    lambda: QApplication.clipboard().setText(name_item.text()))
        copy_all = menu.addAction("Copy entire table")
        copy_all.triggered.connect(self._copy_table)
        menu.exec(self.tbl.viewport().mapToGlobal(pos))


class ArchitecturePage(QWidget):
    """Visualises the layer-group structure as a tree and shows per-group stats."""
    def __init__(self):
        super().__init__()
        self.analysis: Optional[ModelAnalysis] = None
        ly = QVBoxLayout(self); ly.setSpacing(8)

        # Header
        hl = QHBoxLayout()
        hl.addWidget(QLabel("🏗️ Architecture Explorer"))
        self.search = QLineEdit(); self.search.setPlaceholderText("Search layers…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_tree)
        hl.addWidget(self.search, 1)
        ly.addLayout(hl)

        # Split view
        splitter = QSplitter(Qt.Horizontal)

        # Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Layer Group", "Params", "Size", "Dtypes"])
        self.tree.setColumnWidth(0, 300)
        self.tree.setColumnWidth(1, 100)
        self.tree.setColumnWidth(2, 100)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemClicked.connect(self._on_tree_click)
        self.tree.setExpandsOnDoubleClick(True)
        splitter.addWidget(self.tree)

        # Detail panel
        detail = QWidget()
        dl = QVBoxLayout(detail); dl.setSpacing(8)
        self.detail_name = QLabel("Select a layer group")
        self.detail_name.setStyleSheet(f"font-size:16px;font-weight:bold;color:{C.ACC};background:transparent;")
        dl.addWidget(self.detail_name)

        self.detail_table = QTableWidget()
        self.detail_table.setColumnCount(5)
        self.detail_table.setHorizontalHeaderLabels(["Tensor", "Shape", "Dtype", "Params", "Size"])
        self.detail_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.detail_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.detail_table.setSortingEnabled(True)
        self.detail_table.setAlternatingRowColors(True)
        self.detail_table.setStyleSheet(f"QTableWidget{{alternate-background-color:{C.BG};}}")
        dl.addWidget(self.detail_table)

        # Detail stats
        self.detail_stats = QLabel("")
        self.detail_stats.setStyleSheet(f"color:{C.TXT2};font-size:12px;background:transparent;")
        self.detail_stats.setWordWrap(True)
        dl.addWidget(self.detail_stats)

        splitter.addWidget(detail)
        splitter.setSizes([400, 600])
        ly.addWidget(splitter)

    def update_analysis(self, a: ModelAnalysis):
        self.analysis = a
        self._build_tree(a.layer_groups)

    def _build_tree(self, groups: Dict[str, List[TensorInfo]]):
        self.tree.clear()
        fmt_p = ModelAnalysis.fmt_params
        fmt_s = ModelAnalysis.fmt_size
        total_p = self.analysis.total_params if self.analysis else 1

        # Sort groups by params descending
        sorted_groups = sorted(groups.items(), key=lambda x: -sum(t.num_params for t in x[1]))

        for group_name, tensors in sorted_groups:
            total_params = sum(t.num_params for t in tensors)
            total_size = sum(t.size_bytes for t in tensors)
            dtypes = ", ".join(sorted(set(t.dtype for t in tensors)))
            pct = total_params / total_p * 100 if total_p > 0 else 0

            item = QTreeWidgetItem([
                group_name,
                f"{fmt_p(total_params)} ({pct:.1f}%)",
                fmt_s(total_size),
                dtypes
            ])
            item.setData(0, Qt.UserRole, group_name)

            # Color by relative size
            if pct > 10:
                item.setForeground(0, QBrush(QColor(C.ACC)))
            elif pct > 5:
                item.setForeground(0, QBrush(QColor(C.GRN)))

            # Add children for each tensor
            for t in sorted(tensors, key=lambda x: -x.num_params):
                child = QTreeWidgetItem([
                    t.name.split('.')[-1] if '.' in t.name else t.name,
                    fmt_p(t.num_params),
                    fmt_s(t.size_bytes),
                    t.dtype
                ])
                child.setData(0, Qt.UserRole, t.name)
                child.setForeground(2, QBrush(QColor(C.TXT2)))
                item.addChild(child)

            self.tree.addTopLevelItem(item)

    def _on_tree_click(self, item: QTreeWidgetItem, column: int):
        if not self.analysis:
            return
        data = item.data(0, Qt.UserRole)
        if data in self.analysis.layer_groups:
            self._show_group_detail(data, self.analysis.layer_groups[data])
        elif item.parent():
            parent_data = item.parent().data(0, Qt.UserRole)
            if parent_data in self.analysis.layer_groups:
                self._show_group_detail(parent_data, self.analysis.layer_groups[parent_data])

    def _show_group_detail(self, name: str, tensors: List[TensorInfo]):
        fmt_p = ModelAnalysis.fmt_params
        fmt_s = ModelAnalysis.fmt_size

        self.detail_name.setText(name)
        total_p = sum(t.num_params for t in tensors)
        total_s = sum(t.size_bytes for t in tensors)
        dtypes = sorted(set(t.dtype for t in tensors))
        self.detail_stats.setText(
            f"Tensors: {len(tensors)} | Params: {fmt_p(total_p)} | "
            f"Size: {fmt_s(total_s)} | Dtypes: {', '.join(dtypes)}"
        )

        self.detail_table.setRowCount(len(tensors))
        for r, t in enumerate(sorted(tensors, key=lambda x: -x.num_params)):
            it_name = QTableWidgetItem(t.name)
            it_name.setFlags(it_name.flags() & ~Qt.ItemIsEditable)
            it_name.setToolTip(t.name)
            it_shape = QTableWidgetItem(str(t.shape))
            it_shape.setFlags(it_shape.flags() & ~Qt.ItemIsEditable)
            it_dtype = QTableWidgetItem(t.dtype)
            it_dtype.setFlags(it_dtype.flags() & ~Qt.ItemIsEditable)
            it_params = NumericItem(fmt_p(t.num_params), t.num_params)
            it_params.setFlags(it_params.flags() & ~Qt.ItemIsEditable)
            it_size = NumericItem(fmt_s(t.size_bytes), t.size_bytes)
            it_size.setFlags(it_size.flags() & ~Qt.ItemIsEditable)

            self.detail_table.setItem(r, 0, it_name)
            self.detail_table.setItem(r, 1, it_shape)
            self.detail_table.setItem(r, 2, it_dtype)
            self.detail_table.setItem(r, 3, it_params)
            self.detail_table.setItem(r, 4, it_size)

    def _filter_tree(self):
        txt = self.search.text().lower()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            match = not txt or txt in item.text(0).lower()
            # Check children
            for j in range(item.childCount()):
                child = item.child(j)
                if not txt or txt in child.text(0).lower() or txt in item.text(0).lower():
                    match = True
                    break
            item.setHidden(not match)


class MetadataPage(QWidget):
    """Dedicated metadata viewer with search and filtering."""
    def __init__(self):
        super().__init__()
        self.analysis: Optional[ModelAnalysis] = None
        ly = QVBoxLayout(self); ly.setSpacing(8)

        # Header
        hl = QHBoxLayout()
        hl.addWidget(QLabel("📋 Metadata"))
        self.search = QLineEdit(); self.search.setPlaceholderText("Search metadata keys…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        hl.addWidget(self.search, 1)

        self.copy_btn = QPushButton("📋 Copy All")
        self.copy_btn.clicked.connect(self._copy_all)
        hl.addWidget(self.copy_btn)

        ly.addLayout(hl)

        # Table
        self.tbl = QTableWidget()
        self.tbl.setColumnCount(3)
        self.tbl.setHorizontalHeaderLabels(["Key", "Value", "Type"])
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setSortingEnabled(True)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setStyleSheet(f"QTableWidget{{alternate-background-color:{C.BG};}}")

        self.tbl.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl.customContextMenuRequested.connect(self._context_menu)

        ly.addWidget(self.tbl)

    def update_analysis(self, a: ModelAnalysis):
        self.analysis = a
        self._populate(a.metadata)

    def _populate(self, metadata: Dict[str, Any]):
        self.tbl.setSortingEnabled(False)
        self.tbl.setRowCount(len(metadata))
        for r, (key, val) in enumerate(sorted(metadata.items())):
            it_key = QTableWidgetItem(key)
            it_key.setFlags(it_key.flags() & ~Qt.ItemIsEditable)
            it_key.setToolTip(key)

            val_str = str(val)
            if len(val_str) > 500:
                val_str = val_str[:500] + "…"
            it_val = QTableWidgetItem(val_str)
            it_val.setFlags(it_val.flags() & ~Qt.ItemIsEditable)
            it_val.setToolTip(str(val))

            # Type detection
            vtype = type(val).__name__
            if isinstance(val, int):
                vtype = "int"
            elif isinstance(val, float):
                vtype = "float"
            elif isinstance(val, str):
                vtype = "string"
            elif isinstance(val, list):
                vtype = f"list[{len(val)}]"
            elif isinstance(val, dict):
                vtype = f"dict[{len(val)}]"

            it_type = QTableWidgetItem(vtype)
            it_type.setFlags(it_type.flags() & ~Qt.ItemIsEditable)

            self.tbl.setItem(r, 0, it_key)
            self.tbl.setItem(r, 1, it_val)
            self.tbl.setItem(r, 2, it_type)
        self.tbl.setSortingEnabled(True)

    def _filter(self):
        if not self.analysis:
            return
        txt = self.search.text().lower()
        if not txt:
            self._populate(self.analysis.metadata)
            return
        filtered = {k: v for k, v in self.analysis.metadata.items()
                     if txt in k.lower() or txt in str(v).lower()}
        self._populate(filtered)

    def _copy_all(self):
        if not self.analysis:
            return
        lines = []
        for k, v in sorted(self.analysis.metadata.items()):
            lines.append(f"{k}: {v}")
        QApplication.clipboard().setText("\n".join(lines))

    def _context_menu(self, pos: QPoint):
        menu = QMenu(self)
        item = self.tbl.itemAt(pos)
        if item:
            row = item.row()
            key_item = self.tbl.item(row, 0)
            val_item = self.tbl.item(row, 1)
            if key_item:
                copy_key = menu.addAction("Copy key")
                copy_key.triggered.connect(
                    lambda: QApplication.clipboard().setText(key_item.text()))
            if val_item:
                copy_val = menu.addAction("Copy value")
                copy_val.triggered.connect(
                    lambda: QApplication.clipboard().setText(val_item.text()))
        menu.exec(self.tbl.viewport().mapToGlobal(pos))


class MemoryPage(QWidget):
    """Memory calculator and quantization simulator."""
    def __init__(self):
        super().__init__()
        self.analysis: Optional[ModelAnalysis] = None
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        c = QWidget()
        ml = QVBoxLayout(c); ml.setSpacing(16); ml.setContentsMargins(8,8,8,8)

        ml.addWidget(QLabel("💾 VRAM Estimator & Quantization Simulator"))

        # ── Quantization simulator ──────────────────────────────────────
        q_grp = QGroupBox("Quantization Size Estimator")
        qly = QVBoxLayout(q_grp)

        desc = QLabel("See how different quantization levels affect model size.")
        desc.setStyleSheet(f"color:{C.TXT2};background:transparent;")
        desc.setWordWrap(True)
        qly.addWidget(desc)

        # Results table
        self.quant_tbl = QTableWidget()
        self.quant_tbl.setColumnCount(5)
        self.quant_tbl.setHorizontalHeaderLabels(
            ["Quantization", "Model Size", "vs FP32", "vs FP16", "Fits In"])
        self.quant_tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 5):
            self.quant_tbl.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.quant_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.quant_tbl.setAlternatingRowColors(True)
        self.quant_tbl.setStyleSheet(f"QTableWidget{{alternate-background-color:{C.BG};}}")
        qly.addWidget(self.quant_tbl)

        ml.addWidget(q_grp)

        # ── KV Cache & Context estimator ────────────────────────────────
        kv_grp = QGroupBox("KV Cache & Context Length Estimator")
        kvly = QVBoxLayout(kv_grp)

        form = QFormLayout(); form.setSpacing(12)
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(128, 1048576)
        self.ctx_spin.setValue(4096)
        self.ctx_spin.setSingleStep(512)
        self.ctx_spin.setSuffix(" tokens")
        self.ctx_spin.valueChanged.connect(self._update_kv)
        form.addRow("Context Length:", self.ctx_spin)

        self.kv_result = QLabel("—")
        self.kv_result.setStyleSheet(f"font-size:14px;color:{C.GRN};background:transparent;")
        form.addRow("KV Cache Size:", self.kv_result)

        kvly.addLayout(form)

        # Context length presets
        preset_row = QHBoxLayout()
        for name, val in [("1K", 1024), ("2K", 2048), ("4K", 4096),
                          ("8K", 8192), ("16K", 16384), ("32K", 32768),
                          ("64K", 65536), ("128K", 131072)]:
            b = QPushButton(name)
            b.setFixedWidth(55)
            b.clicked.connect(lambda checked, v=val: self.ctx_spin.setValue(v))
            preset_row.addWidget(b)
        preset_row.addStretch()
        kvly.addLayout(preset_row)

        ml.addWidget(kv_grp)

        # ── VRAM estimator ──────────────────────────────────────────────
        vram_grp = QGroupBox("VRAM Requirement Estimator")
        vramly = QVBoxLayout(vram_grp)

        form2 = QFormLayout(); form2.setSpacing(12)

        self.quant_cb = QComboBox()
        for name in QUANT_OPTIONS:
            self.quant_cb.addItem(name)
        self.quant_cb.setCurrentIndex(1)  # FP16 default
        self.quant_cb.currentTextChanged.connect(self._update_vram)
        form2.addRow("Quantization:", self.quant_cb)

        vramly.addLayout(form2)

        self.vram_results = QLabel("—")
        self.vram_results.setStyleSheet(f"font-size:14px;background:transparent;")
        self.vram_results.setWordWrap(True)
        vramly.addWidget(self.vram_results)

        # GPU compatibility
        self.gpu_grp = QGroupBox("GPU Compatibility")
        gpu_ly = QVBoxLayout(self.gpu_grp)
        self.gpu_results = QLabel("—")
        self.gpu_results.setStyleSheet(f"font-size:12px;background:transparent;")
        self.gpu_results.setWordWrap(True)
        gpu_ly.addWidget(self.gpu_results)
        vramly.addWidget(self.gpu_grp)

        ml.addWidget(vram_grp)

        # ── Chart ───────────────────────────────────────────────────────
        if HAS_CHARTS:
            self.vram_chart_view = QChartView()
            self.vram_chart_view.setRenderHint(QPainter.Antialiasing)
            self.vram_chart_view.setMinimumHeight(250)
            self.vram_chart_view.setStyleSheet("background:transparent;")
            ml.addWidget(self.vram_chart_view)

        ml.addStretch()
        scroll.setWidget(c)
        ol = QVBoxLayout(self); ol.setContentsMargins(0,0,0,0); ol.addWidget(scroll)

    def update_analysis(self, a: ModelAnalysis):
        self.analysis = a
        self._update_quant_table()
        self._update_kv()
        self._update_vram()

    def _update_quant_table(self):
        if not self.analysis:
            return
        a = self.analysis
        fmt_s = ModelAnalysis.fmt_size
        total_p = a.total_params
        fp32_size = total_p * 4
        fp16_size = total_p * 2

        self.quant_tbl.setRowCount(len(QUANT_OPTIONS))
        for r, (name, bpp) in enumerate(QUANT_OPTIONS.items()):
            size_bytes = total_p * bpp
            vs_fp32 = size_bytes / fp32_size * 100 if fp32_size else 0
            vs_fp16 = size_bytes / fp16_size * 100 if fp16_size else 0

            it_name = QTableWidgetItem(name)
            it_name.setFlags(it_name.flags() & ~Qt.ItemIsEditable)

            it_size = NumericItem(fmt_s(size_bytes), size_bytes)
            it_size.setFlags(it_size.flags() & ~Qt.ItemIsEditable)

            it_fp32 = NumericItem(f"{vs_fp32:.1f}%", vs_fp32)
            it_fp32.setFlags(it_fp32.flags() & ~Qt.ItemIsEditable)

            it_fp16 = NumericItem(f"{vs_fp16:.1f}%", vs_fp16)
            it_fp16.setFlags(it_fp16.flags() & ~Qt.ItemIsEditable)

            # Find smallest GPU that fits
            fits = ""
            size_gb = size_bytes / (1024 ** 3)
            for gpu_name, vram in GPU_VRAM.items():
                if size_gb <= vram * 0.9:  # 90% utilization
                    fits = gpu_name
                    break
            it_fits = QTableWidgetItem(fits if fits else "❌ Needs >64GB")
            it_fits.setFlags(it_fits.flags() & ~Qt.ItemIsEditable)

            # Color code
            if bpp <= 0.5:
                it_name.setForeground(QBrush(QColor(C.GRN)))
            elif bpp <= 1.0:
                it_name.setForeground(QBrush(QColor(C.TL)))
            elif bpp <= 2.0:
                it_name.setForeground(QBrush(QColor(C.YEL)))
            else:
                it_name.setForeground(QBrush(QColor(C.RED)))

            self.quant_tbl.setItem(r, 0, it_name)
            self.quant_tbl.setItem(r, 1, it_size)
            self.quant_tbl.setItem(r, 2, it_fp32)
            self.quant_tbl.setItem(r, 3, it_fp16)
            self.quant_tbl.setItem(r, 4, it_fits)

        if HAS_CHARTS:
            self._vram_chart()

    def _update_kv(self):
        if not self.analysis:
            return
        ctx = self.ctx_spin.value()
        kv_bytes = self.analysis.estimate_kv_cache(ctx)
        fmt_s = ModelAnalysis.fmt_size
        self.kv_result.setText(fmt_s(kv_bytes))

    def _update_vram(self):
        if not self.analysis:
            return
        a = self.analysis
        ctx = self.ctx_spin.value()
        bpp = QUANT_OPTIONS.get(self.quant_cb.currentText(), 2.0)
        est = a.estimate_vram(ctx, bpp)
        fmt_s = ModelAnalysis.fmt_size

        text = (
            f"<b>Model weights:</b> {fmt_s(est['model_gb'] * 1024**3)} "
            f"({est['model_gb']:.2f} GB)<br>"
            f"<b>KV cache (ctx={ctx:,}):</b> {fmt_s(est['kv_cache_gb'] * 1024**3)} "
            f"({est['kv_cache_gb']:.2f} GB)<br>"
            f"<b>Overhead (~20%):</b> {fmt_s(est['overhead_gb'] * 1024**3)} "
            f"({est['overhead_gb']:.2f} GB)<br>"
            f"<br><b>Total estimated VRAM: {est['total_gb']:.2f} GB</b>"
        )
        self.vram_results.setText(text)

        # GPU compatibility
        total_gb = est['total_gb']
        gpu_lines = []
        for gpu_name, vram in GPU_VRAM.items():
            fits = total_gb <= vram * 0.9
            icon = "✅" if fits else "❌"
            bar_max = vram
            pct = min(total_gb / vram * 100, 100) if vram > 0 else 0
            color = C.GRN if fits else C.RED
            gpu_lines.append(
                f"{icon} <span style='color:{color}'>{gpu_name}</span> "
                f"({vram}GB) — {pct:.0f}% used"
            )
        self.gpu_results.setText("<br>".join(gpu_lines))

    def _vram_chart(self):
        if not self.analysis or not HAS_CHARTS:
            return
        ch = QChart()
        ch.setTitle("Model Size by Quantization")
        ch.setAnimationOptions(QChart.SeriesAnimations)
        ch.layout().setContentsMargins(0,0,0,0)

        total_p = self.analysis.total_params
        bs = QBarSet("Size (GB)")
        cats = []
        colors = [C.RED, C.YEL, C.TL, C.GRN, C.ACC, C.ACC2, C.PNK, C.LAV]
        for name, bpp in list(QUANT_OPTIONS.items())[:10]:
            size_gb = total_p * bpp / (1024 ** 3)
            bs.append(size_gb)
            short = name.split('(')[0].strip()
            cats.append(short)

        series = QBarSeries(); series.append(bs)
        bs.setBrush(QBrush(QColor(C.ACC)))
        ch.addSeries(series)

        ax = QBarCategoryAxis(); ax.append(cats)
        ax.setLabelsAngle(-45); ax.setLabelsColor(QColor(C.TXT2))
        ch.addAxis(ax, Qt.AlignBottom); series.attachAxis(ax)

        ay = QValueAxis(); ay.setLabelsColor(QColor(C.TXT2))
        ay.setTitleText("GB"); ay.setTitleBrush(QBrush(QColor(C.TXT2)))
        ch.addAxis(ay, Qt.AlignLeft); series.attachAxis(ay)

        ch.legend().setVisible(False)
        self.vram_chart_view.setChart(ch)


class ComparisonPage(QWidget):
    """Compare multiple model analyses side by side."""
    def __init__(self):
        super().__init__()
        self.analyses: List[ModelAnalysis] = []
        ly = QVBoxLayout(self); ly.setSpacing(8)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("⚖️ Model Comparison"))
        add_btn = QPushButton("+ Add Model")
        add_btn.clicked.connect(self._add_model)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear)
        hl.addStretch(); hl.addWidget(add_btn); hl.addWidget(clear_btn)
        ly.addLayout(hl)

        # Comparison table
        self.tbl = QTableWidget()
        self.tbl.setHorizontalHeaderLabels(["Property"])
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setStyleSheet(f"QTableWidget{{alternate-background-color:{C.BG};}}")
        ly.addWidget(self.tbl)

        # Chart
        if HAS_CHARTS:
            self.chart_view = QChartView()
            self.chart_view.setRenderHint(QPainter.Antialiasing)
            self.chart_view.setMinimumHeight(250)
            self.chart_view.setStyleSheet("background:transparent;")
            ly.addWidget(self.chart_view)

    def add_analysis(self, a: ModelAnalysis):
        self.analyses.append(a)
        self._refresh()

    def _add_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Add Model for Comparison", "", ParserRegistry.filter())
        if p:
            # This will be handled by the main window
            self.parent().load_file(p)

    def _clear(self):
        self.analyses.clear()
        self._refresh()

    def _refresh(self):
        n = len(self.analyses)
        if n == 0:
            self.tbl.setRowCount(0)
            return

        fmt_p = ModelAnalysis.fmt_params
        fmt_s = ModelAnalysis.fmt_size

        # Update columns
        self.tbl.setColumnCount(n + 1)
        for i, a in enumerate(self.analyses):
            name = a.model_name or os.path.basename(a.file_path)
            self.tbl.setHorizontalHeaderItem(i + 1, QTableWidgetItem(name))
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)

        rows = [
            ("File", lambda a: os.path.basename(a.file_path)),
            ("Format", lambda a: f"{a.format} {a.format_version}"),
            ("Architecture", lambda a: a.architecture),
            ("Total Parameters", lambda a: f"{fmt_p(a.total_params)} ({a.total_params:,})"),
            ("File Size", lambda a: fmt_s(a.file_size)),
            ("Est. Uncompressed", lambda a: fmt_s(a.total_size_bytes)),
            ("Compression", lambda a: f"{a.compression_ratio:.1%}"),
            ("Quantization", lambda a: a.quantization_summary),
            ("Context Length", lambda a: f"{a.context_length:,}" if a.context_length else "—"),
            ("Vocab Size", lambda a: f"{a.vocab_size:,}" if a.vocab_size else "—"),
            ("Hidden Layers", lambda a: str(a.num_hidden_layers) if a.num_hidden_layers else "—"),
            ("Hidden Size", lambda a: str(a.hidden_size) if a.hidden_size else "—"),
            ("Attention Heads", lambda a: str(a.num_attention_heads) if a.num_attention_heads else "—"),
            ("KV Heads", lambda a: str(a.num_kv_heads) if a.num_kv_heads else "—"),
            ("Intermediate Size", lambda a: str(a.intermediate_size) if a.intermediate_size else "—"),
            ("Num Tensors", lambda a: str(len(a.tensors))),
            ("Layer Groups", lambda a: str(len(a.layer_groups))),
            ("Dtypes", lambda a: ", ".join(sorted(a.dtype_distribution.keys()))),
            ("Parse Time", lambda a: f"{a.analysis_duration_ms:.0f}ms" if a.analysis_duration_ms else "—"),
        ]

        self.tbl.setRowCount(len(rows))
        for r, (label, getter) in enumerate(rows):
            it_prop = QTableWidgetItem(label)
            it_prop.setFlags(it_prop.flags() & ~Qt.ItemIsEditable)
            it_prop.setFont(QFont("", -1, QFont.Bold))
            self.tbl.setItem(r, 0, it_prop)

            for i, a in enumerate(self.analyses):
                it_val = QTableWidgetItem(getter(a))
                it_val.setFlags(it_val.flags() & ~Qt.ItemIsEditable)
                self.tbl.setItem(r, i + 1, it_val)

        # Auto-resize columns
        for col in range(1, n + 1):
            self.tbl.horizontalHeader().setSectionResizeMode(col, QHeaderView.Stretch)

        if HAS_CHARTS and n > 0:
            self._comparison_chart()

    def _comparison_chart(self):
        ch = QChart()
        ch.setTitle("Parameter Comparison")
        ch.setAnimationOptions(QChart.SeriesAnimations)
        ch.layout().setContentsMargins(0,0,0,0)

        colors = [C.ACC, C.GRN, C.ORG, C.ACC2, C.TL, C.YEL, C.RED, C.PNK]
        bs = QBarSet("Parameters")
        cats = []
        for i, a in enumerate(self.analyses):
            name = a.model_name or os.path.basename(a.file_path)
            if len(name) > 15:
                name = name[:13] + "…"
            cats.append(name)
            bs.append(a.total_params)

        series = QBarSeries(); series.append(bs)
        bs.setBrush(QBrush(QColor(C.ACC)))
        ch.addSeries(series)

        ax = QBarCategoryAxis(); ax.append(cats)
        ax.setLabelsColor(QColor(C.TXT2))
        ch.addAxis(ax, Qt.AlignBottom); series.attachAxis(ax)

        ay = QValueAxis(); ay.setLabelsColor(QColor(C.TXT2))
        ch.addAxis(ay, Qt.AlignLeft); series.attachAxis(ay)

        ch.legend().setVisible(False)
        self.chart_view.setChart(ch)


# ============================================================================
# Main Window
# ============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Model Analyzer v2.0")
        self.setMinimumSize(1100, 750)
        self.resize(1300, 850)
        self.setAcceptDrops(True)

        self.settings = QSettings("NNAnalyzer", "ModelAnalyzer")
        self.analysis: Optional[ModelAnalysis] = None
        self.worker: Optional[AnalysisWorker] = None
        self.batch_worker: Optional[BatchAnalysisWorker] = None
        self.comparison_analyses: List[ModelAnalysis] = []

        self._build_ui()
        self._build_menu()
        self._build_statusbar()
        self._apply_shortcuts()
        self._restore_geometry()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        ml = QHBoxLayout(central); ml.setSpacing(0); ml.setContentsMargins(0,0,0,0)

        # ── Sidebar ─────────────────────────────────────────────────────
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet(
            f"QFrame{{background:{C.BG2};border-right:1px solid {C.OVER};}}"
        )
        sl = QVBoxLayout(sidebar); sl.setContentsMargins(8,8,8,8); sl.setSpacing(4)

        # Logo
        logo = QLabel("🧠 NN Analyzer")
        logo.setStyleSheet(f"font-size:16px;font-weight:bold;color:{C.ACC};padding:12px 8px;background:transparent;")
        sl.addWidget(logo)

        # Navigation buttons
        self.nav_buttons = []
        recent = self.settings.value("recent_files", [])
        if isinstance(recent, str):
            recent = [recent]

        pages = [
            ("welcome", "🏠  Welcome", "Welcome page"),
            ("overview", "📊  Overview", "Model overview & stats"),
            ("tensors", "🔢  Tensors", "Tensor details & search"),
            ("architecture", "🏗️  Architecture", "Layer group explorer"),
            ("metadata", "📋  Metadata", "Model metadata viewer"),
            ("memory", "💾  Memory", "VRAM & quantization estimator"),
            ("comparison", "⚖️  Compare", "Compare models"),
        ]
        for key, label, tooltip in pages:
            btn = QPushButton(label)
            btn.setObjectName("nav")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda checked, k=key: self._switch_page(k))
            sl.addWidget(btn)
            self.nav_buttons.append((key, btn))

        sl.addStretch()

        # Recent files in sidebar
        if recent:
            sl.addWidget(QLabel("Recent:"))
            for rf in recent[:5]:
                if os.path.exists(rf):
                    btn = QPushButton(f"📄 {os.path.basename(rf)}")
                    btn.setObjectName("nav")
                    btn.setCursor(Qt.PointingHandCursor)
                    btn.setToolTip(rf)
                    btn.clicked.connect(lambda checked, p=rf: self.load_file(p))
                    sl.addWidget(btn)

        # ── Content stack ───────────────────────────────────────────────
        self.stack = QStackedWidget()
        self.pages = {}

        recent_list = self.settings.value("recent_files", [])
        if isinstance(recent_list, str):
            recent_list = [recent_list]

        self.pages["welcome"] = WelcomePage(recent_files=recent_list)
        self.pages["overview"] = OverviewPage()
        self.pages["tensors"] = TensorsPage()
        self.pages["architecture"] = ArchitecturePage()
        self.pages["metadata"] = MetadataPage()
        self.pages["memory"] = MemoryPage()
        self.pages["comparison"] = ComparisonPage()

        for key, page in self.pages.items():
            self.stack.addWidget(page)

        # Connect signals
        self.pages["welcome"].file_dropped.connect(self.load_file)
        self.pages["welcome"].file_selected.connect(self.load_file)
        self.pages["welcome"].batch_requested.connect(self._batch_analysis)

        ml.addWidget(sidebar)
        ml.addWidget(self.stack, stretch=1)

        # Progress overlay
        self.progress_frame = QFrame()
        self.progress_frame.setStyleSheet(f"QFrame{{background:{C.BG};}}")
        ply = QVBoxLayout(self.progress_frame); ply.setAlignment(Qt.AlignCenter); ply.setSpacing(16)
        self.progress_label = QLabel("Analysing…")
        self.progress_label.setStyleSheet(f"font-size:18px;font-weight:bold;color:{C.TXT};background:transparent;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedWidth(400)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_analysis)
        ply.addWidget(self.progress_label, alignment=Qt.AlignCenter)
        ply.addWidget(self.progress_bar, alignment=Qt.AlignCenter)
        ply.addWidget(self.cancel_btn, alignment=Qt.AlignCenter)

        self.stack.addWidget(self.progress_frame)

        self._switch_page("welcome")

    def _build_menu(self):
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu("&File")
        open_act = file_menu.addAction("Open Model…")
        open_act.setShortcut(QKeySequence("Ctrl+O"))
        open_act.triggered.connect(self._open_file)

        batch_act = file_menu.addAction("Batch Analysis…")
        batch_act.setShortcut(QKeySequence("Ctrl+Shift+O"))
        batch_act.triggered.connect(self._batch_analysis)

        file_menu.addSeparator()

        export_act = file_menu.addAction("Export…")
        export_act.setShortcut(QKeySequence("Ctrl+E"))
        export_act.triggered.connect(self._export)

        file_menu.addSeparator()

        close_act = file_menu.addAction("Close Model")
        close_act.setShortcut(QKeySequence("Ctrl+W"))
        close_act.triggered.connect(self._close_model)

        quit_act = file_menu.addAction("Quit")
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)

        # View menu
        view_menu = mb.addMenu("&View")
        for key, label in [("welcome", "Welcome"), ("overview", "Overview"),
                            ("tensors", "Tensors"), ("architecture", "Architecture"),
                            ("metadata", "Metadata"), ("memory", "Memory"),
                            ("comparison", "Compare")]:
            act = view_menu.addAction(label)
            act.triggered.connect(lambda checked, k=key: self._switch_page(k))

        # Tools menu
        tools_menu = mb.addMenu("&Tools")
        hash_act = tools_menu.addAction("Compute File Hashes")
        hash_act.triggered.connect(self._compute_hashes)
        dedup_act = tools_menu.addAction("Find Duplicate Tensors")
        dedup_act.triggered.connect(self._find_duplicates)

        # Help menu
        help_menu = mb.addMenu("&Help")
        about_act = help_menu.addAction("About")
        about_act.triggered.connect(self._show_about)

    def _build_statusbar(self):
        self.statusBar().showMessage("Ready — Drop a model file or press Ctrl+O")
        self.status_model = QLabel("")
        self.status_model.setStyleSheet(f"color:{C.ACC};padding:0 8px;")
        self.statusBar().addPermanentWidget(self.status_model)

    def _apply_shortcuts(self):
        shortcuts = [
            ("Ctrl+1", lambda: self._switch_page("welcome")),
            ("Ctrl+2", lambda: self._switch_page("overview")),
            ("Ctrl+3", lambda: self._switch_page("tensors")),
            ("Ctrl+4", lambda: self._switch_page("architecture")),
            ("Ctrl+5", lambda: self._switch_page("metadata")),
            ("Ctrl+6", lambda: self._switch_page("memory")),
            ("Ctrl+7", lambda: self._switch_page("comparison")),
        ]
        for key, callback in shortcuts:
            s = QShortcut(QKeySequence(key), self)
            s.activated.connect(callback)

    def _restore_geometry(self):
        geo = self.settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)

    def _switch_page(self, key: str):
        page = self.pages.get(key)
        if page:
            idx = self.stack.indexOf(page)
            self.stack.setCurrentIndex(idx)
            # Update nav button states
            for btn_key, btn in self.nav_buttons:
                btn.setProperty("active", btn_key == key)
                btn.style().unpolish(btn)
                btn.style().polish(btn)

    # ── File operations ─────────────────────────────────────────────────
    def _open_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Model", "", ParserRegistry.filter())
        if p:
            self.load_file(p)

    def load_file(self, path: str):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Analysis already in progress")
            return

        if not os.path.exists(path):
            QMessageBox.critical(self, "Error", f"File not found: {path}")
            return

        self._switch_page("progress")
        self.progress_label.setText(f"Analysing {os.path.basename(path)}…")
        self.progress_bar.setValue(0)
        self.statusBar().showMessage(f"Loading {path}…")

        self.worker = AnalysisWorker(path)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_analysis_finished)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.start()

    def _cancel_analysis(self):
        if self.worker:
            self.worker.cancel()
            self.statusBar().showMessage("Analysis cancelled")
        self._switch_page("welcome")

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(msg)

    def _on_analysis_finished(self, a: ModelAnalysis):
        self.analysis = a
        self.status_model.setText(
            f"{a.model_name or os.path.basename(a.file_path)} | "
            f"{a.fmt_params(a.total_params)} | {a.fmt_size(a.file_size)}"
        )
        self.statusBar().showMessage(f"Analysis complete in {a.analysis_duration_ms:.0f}ms")

        # Update all pages
        self.pages["overview"].update_analysis(a)
        self.pages["tensors"].update_analysis(a)
        self.pages["architecture"].update_analysis(a)
        self.pages["metadata"].update_analysis(a)
        self.pages["memory"].update_analysis(a)

        # Add to comparison
        self.pages["comparison"].add_analysis(a)

        # Save to recent
        self._add_recent(a.file_path)

        # Navigate to overview
        self._switch_page("overview")

    def _on_analysis_error(self, msg: str):
        QMessageBox.critical(self, "Analysis Error", msg)
        self._switch_page("welcome")
        self.statusBar().showMessage("Analysis failed")

    def _close_model(self):
        self.analysis = None
        self.status_model.setText("")
        self.statusBar().showMessage("Model closed")
        self._switch_page("welcome")

    def _export(self):
        if not self.analysis:
            QMessageBox.information(self, "No Model", "Load a model first")
            return
        dlg = ExportDialog(self.analysis, self)
        dlg.exec()

    def _batch_analysis(self):
        dlg = BatchDialog(self)
        dlg.files_selected.connect(self._start_batch)
        dlg.exec()

    def _start_batch(self, paths: List[str]):
        if self.batch_worker and self.batch_worker.isRunning():
            return
        self._switch_page("progress")
        self.progress_label.setText("Batch analysis…")
        self.progress_bar.setValue(0)
        self.batch_worker = BatchAnalysisWorker(paths)
        self.batch_worker.progress.connect(self._on_progress)
        self.batch_worker.file_finished.connect(
            lambda i, a: self.pages["comparison"].add_analysis(a))
        self.batch_worker.finished.connect(self._on_batch_finished)
        self.batch_worker.error.connect(
            lambda msg: self.statusBar().showMessage(f"Error: {msg}"))
        self.batch_worker.start()

    def _on_batch_finished(self, results: List[ModelAnalysis]):
        self.statusBar().showMessage(f"Batch analysis complete: {len(results)} models")
        self._switch_page("comparison")

    def _compute_hashes(self):
        if not self.analysis:
            QMessageBox.information(self, "No Model", "Load a model first")
            return
        self.statusBar().showMessage("Computing MD5 hash…")
        md5 = self.analysis.compute_file_hash('md5')
        sha256 = self.analysis.compute_file_hash('sha256')
        self.analysis.metadata['_md5'] = md5
        self.analysis.metadata['_sha256'] = sha256
        self.pages["metadata"].update_analysis(self.analysis)
        self.statusBar().showMessage(f"MD5: {md5}")
        QMessageBox.information(self, "File Hashes",
            f"MD5: {md5}\nSHA256: {sha256}")

    def _find_duplicates(self):
        if not self.analysis:
            QMessageBox.information(self, "No Model", "Load a model first")
            return
        dups = self.analysis.duplicate_tensors
        if dups:
            msg = "\n".join(f"  {n} (×{c})" for n, c in dups)
            QMessageBox.warning(self, "Duplicate Tensors Found", msg)
        else:
            QMessageBox.information(self, "No Duplicates",
                "No duplicate tensor names found.")

    def _add_recent(self, path: str):
        recent = self.settings.value("recent_files", [])
        if isinstance(recent, str):
            recent = [recent]
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        recent = recent[:20]
        self.settings.setValue("recent_files", recent)

    def _show_about(self):
        deps = []
        for name, has in [("PyTorch", HAS_TORCH), ("Safetensors", HAS_SAFETENSORS),
                          ("ONNX", HAS_ONNX), ("HDF5", HAS_H5PY),
                          ("GGUF", HAS_GGUF), ("NumPy", HAS_NUMPY),
                          ("QtCharts", HAS_CHARTS)]:
            deps.append(f"  {'✅' if has else '❌'} {name}")

        QMessageBox.about(self, "About NN Model Analyzer",
            f"🧠 Neural Network Model Analyzer v2.0\n\n"
            f"Analyse LLMs, SLMs, and any neural network models.\n\n"
            f"Supported formats: GGUF, Safetensors, ONNX, PyTorch, HDF5\n\n"
            f"Dependencies:\n" + "\n".join(deps) +
            f"\n\nBuilt with PySide6 (Qt for Python)")

    # ── Drag & drop on main window ──────────────────────────────────────
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        urls = e.mimeData().urls()
        if urls and urls[0].toLocalFile():
            self.load_file(urls[0].toLocalFile())

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.cancel()
            self.batch_worker.wait(3000)
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    app.setStyle("Fusion")

    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()