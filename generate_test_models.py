#!/usr/bin/env python3
"""
Test Model Generator for Neural Network Model Analyzer
Generates 5 dummy model files in different formats to test the analyzer.
"""

import os
import struct
import json

# Create output directory
OUTPUT_DIR = "test_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. PyTorch Model (Full Checkpoint) ---
def generate_pytorch():
    try:
        import torch
        print("1. Generating PyTorch model (.pt)...")
        
        # Create a simple sequential model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )
        
        # Simulate a full training checkpoint (tests the _walk logic)
        checkpoint = {
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {'lr': 0.001},
            'loss': 0.042
        }
        
        path = os.path.join(OUTPUT_DIR, "pytorch_checkpoint.pt")
        torch.save(checkpoint, path)
        print(f"   ✅ Saved: {path}")
    except ImportError:
        print("   ⏭️ Skipping PyTorch (install: pip install torch)")

# --- 2. Safetensors Model (LLaMA-like architecture) ---
def generate_safetensors():
    try:
        import torch
        from safetensors.torch import save_file
        
        print("2. Generating Safetensors model (.safetensors)...")
        
        # Create tensors that mimic an LLM structure (tests architecture inference)
        tensors = {
            "model.embed_tokens.weight": torch.randn(32000, 4096),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(1024, 4096),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(1024, 4096),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(4096, 4096),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(11008, 4096),
            "model.layers.0.mlp.up_proj.weight": torch.randn(11008, 4096),
            "model.layers.0.mlp.down_proj.weight": torch.randn(4096, 11008),
            "lm_head.weight": torch.randn(32000, 4096),
        }
        
        metadata = {
            "model_name": "TestLLaMA-7B",
            "architecture": "llama",
            "context_length": "4096"
        }
        
        path = os.path.join(OUTPUT_DIR, "safetensors_llama.safetensors")
        save_file(tensors, path, metadata=metadata)
        print(f"   ✅ Saved: {path}")
    except ImportError:
        print("   ⏭️ Skipping Safetensors (install: pip install safetensors torch)")

# --- 3. ONNX Model ---
def generate_onnx():
    try:
        import numpy as np
        import onnx
        from onnx import helper, TensorProto
        
        print("3. Generating ONNX model (.onnx)...")
        
        # Create a simple MatMul graph
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 10])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 5])
        
        # Create an initializer (weight matrix)
        W_init = helper.make_tensor(
            'W', 
            TensorProto.FLOAT, 
            [5, 10], 
            np.random.randn(5, 10).astype(np.float32).flatten().tolist()
        )
        
        matmul_node = helper.make_node('MatMul', inputs=['X', 'W'], outputs=['Y'])
        graph = helper.make_graph([matmul_node], 'test_graph', [X], [Y], initializer=[W_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.producer_name = "TestGenerator"
        
        path = os.path.join(OUTPUT_DIR, "onnx_matmul.onnx")
        onnx.save(model, path)
        print(f"   ✅ Saved: {path}")
    except ImportError:
        print("   ⏭️ Skipping ONNX (install: pip install onnx numpy)")

# --- 4. HDF5 Model (Keras-like) ---
def generate_hdf5():
    try:
        import h5py
        import numpy as np
        
        print("4. Generating HDF5 model (.h5)...")
        
        path = os.path.join(OUTPUT_DIR, "hdf5_keras_model.h5")
        with h5py.File(path, 'w') as f:
            # Add root attributes
            f.attrs['model_name'] = 'TestKerasModel'
            f.attrs['keras_version'] = '2.12.0'
            
            # Create nested groups mimicking Keras structure
            grp = f.create_group("model")
            grp.attrs['layer_type'] = 'Sequential'
            
            d1 = grp.create_dataset("dense1_weights", data=np.random.randn(10, 20).astype(np.float32))
            d1.attrs['dtype'] = 'float32'
            
            d2 = grp.create_dataset("dense1_bias", data=np.random.randn(20).astype(np.float32))
            
            grp2 = f.create_group("model/layer2")
            grp2.create_dataset("dense2_weights", data=np.random.randn(20, 5).astype(np.float32))
            grp2.create_dataset("dense2_bias", data=np.random.randn(5).astype(np.float32))
            
        print(f"   ✅ Saved: {path}")
    except ImportError:
        print("   ⏭️ Skipping HDF5 (install: pip install h5py numpy)")

# --- 5. GGUF Model (Manual Binary Write) ---
def generate_gguf():
    print("5. Generating GGUF model (.gguf) - Manual binary construction...")
    
    path = os.path.join(OUTPUT_DIR, "gguf_manual_test.gguf")
    
    def write_string(f, s):
        s_bytes = s.encode('utf-8')
        f.write(struct.pack('<Q', len(s_bytes)))
        f.write(s_bytes)
        
    with open(path, 'wb') as f:
        # Magic: GGUF (0x46475547)
        f.write(struct.pack('<I', 0x46475547))
        # Version: 3
        f.write(struct.pack('<I', 3))
        # n_tensor: 2
        f.write(struct.pack('<Q', 2))
        # n_kv: 2
        f.write(struct.pack('<Q', 2))
        
        # --- Metadata KV 1 ---
        write_string(f, "general.name")
        f.write(struct.pack('<I', 8))  # Type 8 = STRING
        write_string(f, "TestGGUF-Model")
        
        # --- Metadata KV 2 ---
        write_string(f, "general.architecture")
        f.write(struct.pack('<I', 8))  # Type 8 = STRING
        write_string(f, "llama")
        
        # --- Tensor Info 1 ---
        write_string(f, "token_embd.weight")
        f.write(struct.pack('<I', 2))          # ndim = 2
        f.write(struct.pack('<Q', 512))        # dim 0
        f.write(struct.pack('<Q', 256))        # dim 1
        f.write(struct.pack('<I', 0))          # dtype = F32
        f.write(struct.pack('<Q', 0))          # offset
        
        # --- Tensor Info 2 ---
        write_string(f, "output.weight")
        f.write(struct.pack('<I', 2))          # ndim = 2
        f.write(struct.pack('<Q', 256))        # dim 0
        f.write(struct.pack('<Q', 512))        # dim 1
        f.write(struct.pack('<I', 1))          # dtype = F16
        f.write(struct.pack('<Q', 524288))     # offset
        
        # Note: For the analyzer, we only need the header structure. 
        # We don't need to write the actual tensor data payload to test the parser!
        
    print(f"   ✅ Saved: {path}")

if __name__ == "__main__":
    print(f"🧪 Generating test models in ./{OUTPUT_DIR}/ directory...\n")
    generate_pytorch()
    generate_safetensors()
    generate_onnx()
    generate_hdf5()
    generate_gguf()
    print("\n✨ Generation complete! Drag and drop these files into the Analyzer to test.")