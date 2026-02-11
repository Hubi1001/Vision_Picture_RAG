# Punkt 1: Algorytm Porównania Danych ze Skanera z Modelami 3D

## Opis

Moduł `scanner_3d_comparator.py` implementuje kompletny algorytm porównania danych
ze skanera 3D z modelami referencyjnymi z bazy danych. Pipeline identyfikuje model
z biblioteki na podstawie chmury punktów uzyskanej ze skanera.

---

## Architektura systemu

```
┌─────────────────────┐     ┌─────────────────────┐
│   Dane ze skanera   │     │   Baza modeli 3D    │
│   (chmura punktów)  │     │   (STL/OBJ/PLY)     │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│ ScannerDataProcessor│     │ Model3DDatabase      │
│ ├─ Downsampling     │     │ ├─ Deskryptory       │
│ ├─ Denoising        │     │ ├─ Hashowanie        │
│ ├─ Normalizacja PCA │     │ └─ SQLite storage    │
│ └─ Ekstrakcja cech  │     │                     │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           ▼                           ▼
     ┌─────────────────────────────────────┐
     │      Model3DComparator              │
     │  ┌─────────────────────────────┐    │
     │  │ Faza 1: Coarse Matching     │    │
     │  │ (Euclidean + Cosine sim.)   │    │
     │  └──────────┬──────────────────┘    │
     │             ▼                       │
     │  ┌─────────────────────────────┐    │
     │  │ Faza 2: Fine Matching       │    │
     │  │ (Multi-ICP + Hausdorff +    │    │
     │  │  Chamfer distance)          │    │
     │  └──────────┬──────────────────┘    │
     │             ▼                       │
     │       Ranking wyników               │
     └─────────────────────────────────────┘
```

---

## Pipeline przetwarzania

### 1. Przetwarzanie danych skanera (`ScannerDataProcessor`)

| Etap | Metoda | Opis |
|------|--------|------|
| **Downsampling** | Siatka voxelowa | Redukcja gęstości z zachowaniem kształtu (cel: 5000 pkt) |
| **Denoising** | Statistical Outlier Removal | Usuwanie szumu (≈5% outlierów, k=15 sąsiadów) |
| **Normalizacja** | PCA + unit sphere | Centroid → origin, orientacja PCA, skalowanie do sfery |
| **Deskryptor** | Wektor cech 136D | Cechy globalne (8D) + histogramy kształtu (128D) |

### 2. Deskryptor geometryczny (136D)

**Cechy globalne (8D):**
- Compactness (zwartość)
- Sphericity (sferyczność)
- Elongation (wydłużenie)
- Bounding Box ratios (2 proporcje)
- Inertia eigenvalue ratios (3 proporcje)

**Dystrybucje kształtu (128D):**
- D2 histogram (64 bins) — rozkład odległości par punktów
- A3 histogram (64 bins) — rozkład kątów trójek punktów

### 3. Coarse matching

Szybkie wyszukanie kandydatów z bazy (bez geometrii):

```
Score = 0.7 × EuclideanSim(global_8D) + 0.3 × CosineSim(hist_128D)
```

### 4. Fine matching (ICP + metryki)

Precyzyjne dopasowanie geometrii dla top-K kandydatów:

- **Multi-orientation ICP** — 4 początkowe orientacje (kompensacja PCA ambiguity)
- **Hausdorff distance** — worst-case odchylenie
- **Chamfer distance** — średnie odchylenie dwukierunkowe

```
FinalScore = 0.50 × DescriptorSim + 0.20 × ICPscore + 0.10 × HDscore + 0.20 × CDscore
```

---

## Wyniki walidacji

Testowane na 10 modelach z szumem i brakami danych (uśrednione z 5 prób):

| Poziom szumu | Noise σ | Missing % | Top-1 Accuracy | Top-3 Accuracy |
|-------------|---------|-----------|----------------|----------------|
| Niski       | 1%      | 5%        | ~72%           | **100%**       |
| Średni      | 2%      | 10%       | ~66%           | **100%**       |
| Wysoki      | 3%      | 15%       | ~68%           | **96-100%**    |

### Modele testowe

| ID | Typ | Kształt |
|----|-----|---------|
| BOLT-M8x20 | Śruba | Cylinder + heksagon |
| BOLT-M6x16 | Śruba | Cylinder + heksagon |
| NUT-M8 | Nakrętka | Heksagon z otworem |
| BRG-6205 | Łożysko | Pierścień |
| SHAFT-12x300 | Wał | Długi cylinder |
| PIN-6x30 | Kołek | Krótki cylinder |
| SLEEVE-10x15x20 | Tuleja | Cylinder z otworem |
| WASHER-M8 | Podkładka | Płaski pierścień |
| PLATE-100x100x10 | Płyta | Prostopadłościan |
| SPHERE-30 | Kula | Sfera |

---

## Użycie

### Szybka identyfikacja

```python
from scanner_3d_comparator import identify_from_scan

results = identify_from_scan(
    scan_file="scan.ply",
    db_path="models_3d.db"
)

print(f"Najlepsze dopasowanie: {results[0]['model_id']}")
print(f"Score: {results[0]['final_score']:.4f}")
```

### Pełny pipeline

```python
from scanner_3d_comparator import (
    Model3DDatabase,
    Model3DComparator,
    ScannerDataProcessor,
)

# 1. Baza danych
db = Model3DDatabase("models_3d.db")

# 2. Komparator
comparator = Model3DComparator(database=db, verbose=True)

# 3. Identyfikacja
results = comparator.identify(
    scan_points=cloud,  # numpy array (N, 3)
    top_k=10,
    refine_top=8,
)
```

### Porównanie dwóch chmur

```python
metrics = comparator.compare_two_meshes(
    cloud_a, cloud_b,
    label_a="Skan A", label_b="Model B"
)
print(f"ICP RMSE: {metrics['icp_rmse']:.6f}")
print(f"Hausdorff: {metrics['hausdorff_distance']:.6f}")
print(f"Chamfer: {metrics['chamfer_distance']:.6f}")
```

---

## Pliki

| Plik | Opis |
|------|------|
| `scanner_3d_comparator.py` | Moduł algorytmu (~2100 linii) |
| `models_3d.db` | Baza danych SQLite z deskryptorami |
| `demo_3d_models/` | 10 modeli STL do demonstracji |
| `metal_parts_rag.ipynb` | Notebook z demo pipeline (Kroki 1-8) |

---

## Zależności

- `trimesh` — ładowanie mesh (STL/OBJ/PLY), generowanie primitywów
- `scipy` — KDTree, SVD, ConvexHull, Hausdorff distance
- `numpy` — operacje numeryczne
- `sqlalchemy` — ORM dla bazy danych SQLite

---

## Limity i potencjalne ulepszenia

1. **Podobne kształty** — trudności z rozróżnieniem NUT↔BRG (oba pierścienie), PLATE↔WASHER (obie płaskie). Rozwiązanie: dodanie cech curvature.
2. **Skalowanie** — po normalizacji PCA informacja o rozmiarze jest tracona. Rozwiązanie: dodanie raw bounding box dimensions jako cechy.
3. **Rotational symmetry** — cylindry symetryczne mogą mieć niejednoznaczną orientację PCA. Rozwiązanie: więcej startowych orientacji ICP.
4. **Wydajność** — ICP O(n²) per kandydat. Rozwiązanie: KD-tree batch queries, GPU acceleration.
