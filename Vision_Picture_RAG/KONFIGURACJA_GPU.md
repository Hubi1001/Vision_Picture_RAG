# 🚀 Konfiguracja GPU dla Vision RAG

## Status aktualny projektu

Projekt **częściowo** korzysta z GPU. Wprowadzone poprawki:

### ✅ Co już działa:
1. **Model embeddingów (CLIP)** - automatycznie używa GPU gdy dostępne
2. **Diagnostyka GPU** - pokazuje dostępność i parametry
3. **Optymalizacja LLM** - dodano FP16 i device_map="auto"

### 🔧 Co zostało poprawione:

#### 1. Model LLM (Phi-3)
**Przed:**
```python
llm_model.to(DEVICE)  # ❌ BŁĄD - brak przypisania
```

**Po:**
```python
if DEVICE == "cuda":
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, 
        torch_dtype=torch.float16,  # ✅ FP16 - 2x szybciej
        device_map="auto",           # ✅ Automatyczny podział na GPU
        low_cpu_mem_usage=True
    )
else:
    llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
    llm_model = llm_model.to(DEVICE)
```

#### 2. Diagnostyka GPU
Dodana komórka diagnostyczna w [metal_parts_rag.ipynb](metal_parts_rag.ipynb#L82-L115):
```python
print(f"✓ CUDA dostępne: {torch.cuda.is_available()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ Pamięć GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

#### 3. Test GPU
Nowa komórka testowa weryfikuje obliczenia na GPU:
- Mnożenie macierzy 1000x1000
- Pomiar czasu wykonania
- Sprawdzenie zajętej pamięci

---

## 📋 Instalacja PyTorch z CUDA

### Windows/Linux (GPU NVIDIA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### MacOS (bez CUDA):
```bash
pip install torch torchvision
```

### Weryfikacja instalacji:
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Brak'}")
```

---

## 🎯 Jak uruchomić projekt z GPU

### 1. Sprawdź dostępność GPU
Uruchom komórkę 1 w notebooku [metal_parts_rag.ipynb](metal_parts_rag.ipynb):
```
✓ CUDA dostępne: True
✓ GPU: NVIDIA GeForce RTX 3080
✓ Pamięć GPU: 10.0 GB
```

### 2. Uruchom test GPU
Wykonaj nową komórkę testową:
```
✓ Test obliczeniowy GPU: 2.45 ms
✓ Tensor na GPU: cuda:0
✅ GPU działa poprawnie!
```

### 3. Załaduj modele
Modele automatycznie wykorzystają GPU:
- **CLIP**: `embedding_model = SentenceTransformer(..., device='cuda')`
- **Phi-3**: `device_map="auto"` + `torch_dtype=torch.float16`

---

## ⚡ Optymalizacje GPU

### 1. Mixed Precision (FP16)
```python
# Automatyczne w Phi-3
torch_dtype=torch.float16  # 2x szybciej, 50% mniej pamięci
```

### 2. Batch Processing
```python
# Przetwarzanie wielu obrazów naraz
embeddings = embedding_model.encode(images, batch_size=32)
```

### 3. Gradient Accumulation
```python
# Dla dużych modeli przy małej pamięci GPU
optimizer.zero_grad()
for batch in batches:
    loss = model(batch)
    loss.backward()
optimizer.step()
```

---

## 📊 Oczekiwane przyspieszenie

| Operacja | CPU (i7) | GPU (RTX 3080) | Przyspieszenie |
|----------|----------|----------------|----------------|
| CLIP embedding | 150 ms | 15 ms | **10x** |
| Phi-3 generacja | 8 s | 800 ms | **10x** |
| Batch 32 obrazów | 4.8 s | 300 ms | **16x** |

---

## 🐛 Troubleshooting

### Błąd: "CUDA out of memory"
```python
# Zmniejsz batch size
batch_size = 8  # zamiast 32

# Wyczyść cache GPU
torch.cuda.empty_cache()

# Użyj CPU dla dużych modeli
DEVICE = "cpu"
```

### Błąd: "No CUDA GPUs are available"
```bash
# Sprawdź sterowniki NVIDIA
nvidia-smi

# Przeinstaluj PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Wolne działanie mimo GPU
```python
# Upewnij się, że tensory są na GPU
inputs = inputs.to('cuda')  # ✅
outputs = model(inputs)     # GPU

# BŁĄD: tensory na CPU
inputs = inputs  # ❌ domyślnie CPU
```

---

## 📈 Monitoring GPU

```python
# Podczas działania
import torch

print(f"Zajęta: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"Zarezerwowana: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
print(f"Maksymalna: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

# Reset statystyk
torch.cuda.reset_peak_memory_stats()
```

### nvidia-smi (Windows/Linux)
```bash
# Monitoring w czasie rzeczywistym
watch -n 1 nvidia-smi

# Pojedyncze sprawdzenie
nvidia-smi
```

---

## ✅ Checklist konfiguracji GPU

- [ ] Zainstalowany PyTorch z CUDA: `torch.cuda.is_available() == True`
- [ ] Uruchomiony test GPU: `komórka testowa pokazuje cuda:0`
- [ ] Model embeddingów na GPU: `embedding_model.device == 'cuda'`
- [ ] Model LLM z FP16: `llm_model.dtype == torch.float16`
- [ ] Tensory przenoszone na GPU: `.to(DEVICE)` w kodzie
- [ ] Brak błędów CUDA OOM

---

## 🎓 Dodatkowe zasoby

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [Hugging Face GPU Training](https://huggingface.co/docs/transformers/perf_train_gpu_one)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
