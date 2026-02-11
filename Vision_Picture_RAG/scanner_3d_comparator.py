"""
Scanner 3D Comparator — Algorytm porównywania danych ze skanera z modelami 3D
================================================================================

Moduł realizujący punkt 1 zadania:
"Opracowanie algorytmu porównania danych ze skanera z modelami 3D z bazy danych
 – identyfikacja modeli z biblioteki."

Główne komponenty:
  1. Model3DDescriptor  — deskryptor geometryczny modelu 3D
  2. Model3DDatabase    — lokalna baza SQLite z modelami referencyjnymi
  3. ScannerDataProcessor — przetwarzanie chmury punktów ze skanera
  4. Model3DComparator  — algorytm porównywania i identyfikacji

Obsługiwane formaty:
  - Modele 3D: STL, OBJ, PLY, OFF, 3MF, STEP (via trimesh)
  - Dane skanera: PLY, PCD, XYZ, CSV (chmury punktów)

Autor: Vision Picture RAG Team
Data: 2025-12-10 / aktualizacja 2026-02-11
"""

from __future__ import annotations

import json
import hashlib
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from scipy.spatial import KDTree, ConvexHull
from scipy.spatial.distance import directed_hausdorff
from scipy.linalg import svd

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    warnings.warn("trimesh nie jest zainstalowany. pip install trimesh")

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text,
    DateTime, JSON as SA_JSON, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ============================================================================
# 1. DESKRYPTOR GEOMETRYCZNY MODELU 3D
# ============================================================================

Base3D = declarative_base()


@dataclass
class Model3DDescriptor:
    """
    Deskryptor geometryczny modelu 3D — wektor cech opisujący kształt.

    Cechy globalne:
        - volume               : objętość (cm³)
        - surface_area          : pole powierzchni (cm²)
        - bounding_box          : wymiary [dx, dy, dz] (mm)
        - centroid              : środek geometryczny [x, y, z]
        - compactness           : V / (4/3·π·r³) — zwartość kształtu
        - sphericity            : π^(1/3)·(6V)^(2/3) / A
        - elongation            : stosunek osi głównych
        - inertia_eigenvalues   : wartości własne tensora bezwładności

    Dystrybucje kształtu:
        - d2_histogram          : histogram odległości par punktów (64 bins)
        - a3_histogram          : histogram kątów trójek punktów (64 bins)

    Identyfikacja:
        - model_hash            : SHA-256 hash geometrii (fingerprint)
    """

    # Identyfikatory
    model_id: str = ""
    model_name: str = ""
    file_path: str = ""
    model_hash: str = ""

    # Cechy globalne
    volume: float = 0.0
    surface_area: float = 0.0
    bounding_box: List[float] = field(default_factory=lambda: [0, 0, 0])
    centroid: List[float] = field(default_factory=lambda: [0, 0, 0])
    num_vertices: int = 0
    num_faces: int = 0

    # Cechy kształtu
    compactness: float = 0.0
    sphericity: float = 0.0
    elongation: float = 0.0
    inertia_eigenvalues: List[float] = field(default_factory=lambda: [0, 0, 0])

    # Dystrybucje kształtu (histogramy)
    d2_histogram: List[float] = field(default_factory=list)  # 64 bins
    a3_histogram: List[float] = field(default_factory=list)  # 64 bins

    # Metadane
    category: str = ""
    material: str = ""
    tags: List[str] = field(default_factory=list)
    source: str = ""  # "cad", "scan", "generated"

    def to_feature_vector(self) -> np.ndarray:
        """
        Konwersja do płaskiego wektora cech do porównywania.
        Wektor znormalizowany — invariant vs skala.
        """
        global_features = [
            self.compactness,
            self.sphericity,
            self.elongation,
        ]
        # Normalized bounding box ratios
        bb = sorted(self.bounding_box, reverse=True)
        if bb[0] > 0:
            bb_ratios = [bb[1] / bb[0], bb[2] / bb[0]]
        else:
            bb_ratios = [0, 0]
        global_features.extend(bb_ratios)

        # Normalized inertia eigenvalues
        eigs = np.array(self.inertia_eigenvalues)
        eig_sum = np.sum(eigs)
        if eig_sum > 0:
            eigs_norm = (eigs / eig_sum).tolist()
        else:
            eigs_norm = [0, 0, 0]
        global_features.extend(eigs_norm)

        # D2 + A3 histograms (normalized)
        d2 = np.array(self.d2_histogram) if self.d2_histogram else np.zeros(64)
        a3 = np.array(self.a3_histogram) if self.a3_histogram else np.zeros(64)

        d2_sum = np.sum(d2)
        a3_sum = np.sum(a3)
        if d2_sum > 0:
            d2 = d2 / d2_sum
        if a3_sum > 0:
            a3 = a3 / a3_sum

        feature_vec = np.concatenate([global_features, d2, a3])
        return feature_vec.astype(np.float32)

    def to_global_feature_vector(self) -> np.ndarray:
        """
        Wektor cech globalnych (8D) — do szybkiego dopasowania klas kształtów.
        Skuteczniejszy niż pełny wektor do wstępnego przesiewu.
        """
        bb = sorted(self.bounding_box, reverse=True)
        if bb[0] > 0:
            bb_ratios = [bb[1] / bb[0], bb[2] / bb[0]]
        else:
            bb_ratios = [0, 0]

        eigs = np.array(self.inertia_eigenvalues)
        eig_sum = np.sum(eigs)
        if eig_sum > 0:
            eigs_norm = (eigs / eig_sum).tolist()
        else:
            eigs_norm = [0, 0, 0]

        features = [
            self.compactness,
            self.sphericity,
            self.elongation,
            bb_ratios[0],
            bb_ratios[1],
            eigs_norm[0],
            eigs_norm[1],
            eigs_norm[2],
        ]
        return np.array(features, dtype=np.float32)

    def to_histogram_vector(self) -> np.ndarray:
        """Wektor histogramów kształtu (128D) — D2 + A3."""
        d2 = np.array(self.d2_histogram) if self.d2_histogram else np.zeros(64)
        a3 = np.array(self.a3_histogram) if self.a3_histogram else np.zeros(64)
        d2_sum = np.sum(d2)
        a3_sum = np.sum(a3)
        if d2_sum > 0:
            d2 = d2 / d2_sum
        if a3_sum > 0:
            a3 = a3 / a3_sum
        return np.concatenate([d2, a3]).astype(np.float32)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Model3DDescriptor":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================================
# 2. BAZA DANYCH MODELI 3D (SQLAlchemy + SQLite)
# ============================================================================

class Model3DRecord(Base3D):
    """Rekord modelu 3D w bazie danych."""
    __tablename__ = "models_3d"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(100), unique=True, nullable=False, index=True)
    model_name = Column(String(255), nullable=False)
    file_path = Column(Text, default="")
    model_hash = Column(String(64), default="", index=True)
    category = Column(String(100), default="")
    material = Column(String(100), default="")
    tags = Column(SA_JSON, default=list)
    source = Column(String(50), default="cad")

    # Cechy globalne
    volume = Column(Float, default=0.0)
    surface_area = Column(Float, default=0.0)
    bounding_box = Column(SA_JSON, default=list)
    centroid = Column(SA_JSON, default=list)
    num_vertices = Column(Integer, default=0)
    num_faces = Column(Integer, default=0)

    # Cechy kształtu
    compactness = Column(Float, default=0.0)
    sphericity = Column(Float, default=0.0)
    elongation = Column(Float, default=0.0)
    inertia_eigenvalues = Column(SA_JSON, default=list)

    # Dystrybucje kształtu
    d2_histogram = Column(SA_JSON, default=list)
    a3_histogram = Column(SA_JSON, default=list)

    # Feature vector cache (serializowany)
    feature_vector = Column(Text, default="")

    # Metadane
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_descriptor(self) -> Model3DDescriptor:
        return Model3DDescriptor(
            model_id=self.model_id,
            model_name=self.model_name,
            file_path=self.file_path or "",
            model_hash=self.model_hash or "",
            volume=self.volume or 0.0,
            surface_area=self.surface_area or 0.0,
            bounding_box=self.bounding_box or [0, 0, 0],
            centroid=self.centroid or [0, 0, 0],
            num_vertices=self.num_vertices or 0,
            num_faces=self.num_faces or 0,
            compactness=self.compactness or 0.0,
            sphericity=self.sphericity or 0.0,
            elongation=self.elongation or 0.0,
            inertia_eigenvalues=self.inertia_eigenvalues or [0, 0, 0],
            d2_histogram=self.d2_histogram or [],
            a3_histogram=self.a3_histogram or [],
            category=self.category or "",
            material=self.material or "",
            tags=self.tags or [],
            source=self.source or "",
        )


class ComparisonLog(Base3D):
    """Log porównań skanów z modelami."""
    __tablename__ = "comparison_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_source = Column(Text, default="")
    matched_model_id = Column(String(100), default="")
    similarity_score = Column(Float, default=0.0)
    hausdorff_distance = Column(Float, default=-1.0)
    icp_fitness = Column(Float, default=-1.0)
    icp_rmse = Column(Float, default=-1.0)
    method = Column(String(50), default="descriptor")
    num_candidates = Column(Integer, default=0)
    processing_time_ms = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.now)
    details = Column(SA_JSON, default=dict)


class Model3DDatabase:
    """
    Lokalna baza danych modeli 3D z deskryptorami geometrycznymi.

    Użycie:
        db = Model3DDatabase("models_3d.db")
        db.add_model(descriptor)
        results = db.search(query_descriptor, top_k=5)
    """

    def __init__(self, db_path: str = "models_3d.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base3D.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def add_model(self, descriptor: Model3DDescriptor) -> int:
        """Dodaj model 3D do bazy z deskryptorem geometrycznym."""
        with self.SessionLocal() as session:
            # Sprawdź duplikaty po model_id
            existing = session.query(Model3DRecord).filter_by(
                model_id=descriptor.model_id
            ).first()
            if existing:
                # Aktualizuj
                for attr in [
                    "model_name", "file_path", "model_hash", "category",
                    "material", "tags", "source", "volume", "surface_area",
                    "bounding_box", "centroid", "num_vertices", "num_faces",
                    "compactness", "sphericity", "elongation",
                    "inertia_eigenvalues", "d2_histogram", "a3_histogram",
                ]:
                    setattr(existing, attr, getattr(descriptor, attr))
                fv = descriptor.to_feature_vector()
                existing.feature_vector = json.dumps(fv.tolist())
                existing.updated_at = datetime.now()
                session.commit()
                return existing.id

            record = Model3DRecord(
                model_id=descriptor.model_id,
                model_name=descriptor.model_name,
                file_path=descriptor.file_path,
                model_hash=descriptor.model_hash,
                category=descriptor.category,
                material=descriptor.material,
                tags=descriptor.tags,
                source=descriptor.source,
                volume=descriptor.volume,
                surface_area=descriptor.surface_area,
                bounding_box=descriptor.bounding_box,
                centroid=descriptor.centroid,
                num_vertices=descriptor.num_vertices,
                num_faces=descriptor.num_faces,
                compactness=descriptor.compactness,
                sphericity=descriptor.sphericity,
                elongation=descriptor.elongation,
                inertia_eigenvalues=descriptor.inertia_eigenvalues,
                d2_histogram=descriptor.d2_histogram,
                a3_histogram=descriptor.a3_histogram,
                feature_vector=json.dumps(
                    descriptor.to_feature_vector().tolist()
                ),
                created_at=datetime.now(),
            )
            session.add(record)
            session.commit()
            return record.id

    def search(
        self,
        query_descriptor: Model3DDescriptor,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        method: str = "cosine",
    ) -> List[Tuple[Model3DDescriptor, float]]:
        """
        Wyszukaj najlepiej pasujące modele na podstawie deskryptora.

        Scoring odbywa się w dwóch etapach:
        1. Global features (8D) — kształt ogólny (waga 0.6)
        2. Histogram features (128D) — rozkłady kształtu (waga 0.4)

        Args:
            query_descriptor: Deskryptor zapytania (z danych skanera)
            top_k: Ile wyników zwrócić
            category_filter: Opcjonalny filtr kategorii
            method: Metoda porównania ("cosine", "euclidean", "combined")

        Returns:
            Lista (deskryptor, score) posortowana od najlepszego
        """
        query_global = query_descriptor.to_global_feature_vector()
        query_hist = query_descriptor.to_histogram_vector()

        with self.SessionLocal() as session:
            q = session.query(Model3DRecord)
            if category_filter:
                q = q.filter(Model3DRecord.category == category_filter)
            records = q.all()

            if not records:
                return []

            results = []
            for rec in records:
                desc = rec.to_descriptor()
                db_global = desc.to_global_feature_vector()
                db_hist = desc.to_histogram_vector()

                # Euclidean distance na global features (lepiej rozróżnia proporcje)
                global_dist = np.linalg.norm(query_global - db_global)
                global_sim = 1.0 / (1.0 + global_dist * 5)

                # Cosine similarity na histogramach (lepsza dla rozkładów)
                hist_sim = self._cosine_similarity(query_hist, db_hist)

                # Composite score: global features matter more for shape class
                score = 0.7 * global_sim + 0.3 * hist_sim

                results.append((desc, float(score)))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    def search_by_hash(self, model_hash: str) -> Optional[Model3DDescriptor]:
        """Wyszukaj model po hashu geometrii — dokładne dopasowanie."""
        with self.SessionLocal() as session:
            rec = session.query(Model3DRecord).filter_by(
                model_hash=model_hash
            ).first()
            if rec:
                return rec.to_descriptor()
            return None

    def list_models(
        self, category: Optional[str] = None
    ) -> List[Model3DDescriptor]:
        """Lista wszystkich modeli w bazie."""
        with self.SessionLocal() as session:
            q = session.query(Model3DRecord)
            if category:
                q = q.filter(Model3DRecord.category == category)
            return [r.to_descriptor() for r in q.all()]

    def log_comparison(
        self,
        scan_source: str,
        matched_model_id: str,
        similarity_score: float,
        method: str = "descriptor",
        **kwargs,
    ):
        """Zaloguj wynik porównania do bazy."""
        with self.SessionLocal() as session:
            log = ComparisonLog(
                scan_source=scan_source,
                matched_model_id=matched_model_id,
                similarity_score=similarity_score,
                method=method,
                hausdorff_distance=kwargs.get("hausdorff_distance", -1.0),
                icp_fitness=kwargs.get("icp_fitness", -1.0),
                icp_rmse=kwargs.get("icp_rmse", -1.0),
                num_candidates=kwargs.get("num_candidates", 0),
                processing_time_ms=kwargs.get("processing_time_ms", 0.0),
                details=kwargs.get("details", {}),
                timestamp=datetime.now(),
            )
            session.add(log)
            session.commit()
            return log.id

    def get_comparison_history(self, limit: int = 20) -> List[dict]:
        """Pobierz historię porównań."""
        with self.SessionLocal() as session:
            logs = (
                session.query(ComparisonLog)
                .order_by(ComparisonLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": l.id,
                    "scan_source": l.scan_source,
                    "matched_model_id": l.matched_model_id,
                    "similarity_score": l.similarity_score,
                    "hausdorff_distance": l.hausdorff_distance,
                    "icp_fitness": l.icp_fitness,
                    "icp_rmse": l.icp_rmse,
                    "method": l.method,
                    "processing_time_ms": l.processing_time_ms,
                    "timestamp": str(l.timestamp),
                }
                for l in logs
            ]

    def count(self) -> int:
        with self.SessionLocal() as session:
            return session.query(Model3DRecord).count()

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================================
# 3. PRZETWARZANIE DANYCH ZE SKANERA (chmury punktów)
# ============================================================================


class ScannerDataProcessor:
    """
    Przetwarzanie danych ze skanera 3D (chmury punktów).

    Obsługuje formaty:
        - PLY (Stanford Point Cloud)
        - XYZ / CSV (x,y,z na wiersz)
        - STL (konwersja z mesh to points)
        - PCD (Point Cloud Data — uproszczony parser)
        - numpy array (programowe)

    Pipeline:
        load → downsample → denoise → normalize → extract_features
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.points: Optional[np.ndarray] = None
        self.normals: Optional[np.ndarray] = None
        self.source_path: str = ""

    def load(self, source) -> "ScannerDataProcessor":
        """
        Załaduj chmurę punktów z pliku lub numpy array.

        Args:
            source: Ścieżka do pliku (str/Path) lub numpy array (N, 3)
        """
        if isinstance(source, np.ndarray):
            self.points = source.copy()
            self.source_path = "<numpy_array>"
            if self.verbose:
                print(f"📥 Załadowano {len(self.points)} punktów z numpy array")
            return self

        source = str(source)
        self.source_path = source
        ext = os.path.splitext(source)[1].lower()

        if ext in (".stl", ".obj", ".ply", ".off", ".3mf"):
            self._load_mesh_file(source)
        elif ext in (".xyz", ".csv", ".txt", ".asc"):
            self._load_xyz_file(source)
        elif ext == ".pcd":
            self._load_pcd_file(source)
        elif ext == ".npy":
            self.points = np.load(source)
            if self.verbose:
                print(f"📥 Załadowano {len(self.points)} punktów z .npy")
        else:
            raise ValueError(f"Nieobsługiwany format: {ext}")

        return self

    def _load_mesh_file(self, path: str):
        """Załaduj mesh i próbkuj punkty z powierzchni."""
        if not TRIMESH_AVAILABLE:
            raise ImportError("Wymagany pakiet: pip install trimesh")
        mesh = trimesh.load(path, force="mesh")
        # Próbkuj 10000 punktów z powierzchni
        n_samples = min(10000, len(mesh.vertices) * 3)
        self.points = np.array(
            mesh.sample(n_samples), dtype=np.float64
        )
        if hasattr(mesh, "vertex_normals") and len(mesh.vertex_normals) > 0:
            # Próbkuj normalne w przybliżeniu
            pass
        if self.verbose:
            print(
                f"📥 Załadowano mesh: {len(mesh.vertices)} wierzchołków, "
                f"{len(mesh.faces)} ścian → {len(self.points)} punktów"
            )

    def _load_xyz_file(self, path: str):
        """Załaduj chmurę punktów z pliku XYZ/CSV."""
        points = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                parts = line.replace(",", " ").split()
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
                except (IndexError, ValueError):
                    continue
        self.points = np.array(points, dtype=np.float64)
        if self.verbose:
            print(f"📥 Załadowano {len(self.points)} punktów z {path}")

    def _load_pcd_file(self, path: str):
        """Uproszczony parser formatu PCD."""
        points = []
        header_done = False
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not header_done:
                    if line.startswith("DATA"):
                        header_done = True
                    continue
                parts = line.split()
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
                except (IndexError, ValueError):
                    continue
        self.points = np.array(points, dtype=np.float64)
        if self.verbose:
            print(f"📥 Załadowano {len(self.points)} punktów z PCD")

    def downsample(self, target_points: int = 5000) -> "ScannerDataProcessor":
        """
        Równomierne próbkowanie chmury punktów (voxel grid downsampling).
        """
        if self.points is None or len(self.points) == 0:
            return self
        if len(self.points) <= target_points:
            return self

        # Voxel grid downsampling
        pts = self.points
        bb_min = pts.min(axis=0)
        bb_max = pts.max(axis=0)
        bb_range = bb_max - bb_min
        max_range = np.max(bb_range)

        if max_range == 0:
            return self

        # Dobranie rozmiaru voxela iteracyjnie
        voxel_size = max_range / 50.0
        for _ in range(20):
            voxel_indices = np.floor((pts - bb_min) / voxel_size).astype(int)
            unique_keys = {}
            for i, idx in enumerate(voxel_indices):
                key = tuple(idx)
                if key not in unique_keys:
                    unique_keys[key] = []
                unique_keys[key].append(i)

            n_voxels = len(unique_keys)
            if n_voxels <= target_points:
                break
            voxel_size *= 1.3

        # Średni punkt w każdym voxelu
        new_points = []
        for indices in unique_keys.values():
            new_points.append(pts[indices].mean(axis=0))

        self.points = np.array(new_points, dtype=np.float64)

        if self.verbose:
            print(
                f"🔽 Downsampling: {len(pts)} → {len(self.points)} punktów "
                f"(voxel={voxel_size:.3f})"
            )
        return self

    def denoise(self, nb_neighbors: int = 20, std_ratio: float = 2.0) -> "ScannerDataProcessor":
        """
        Usunięcie szumu statystycznego (Statistical Outlier Removal).
        Punkty dalsze niż std_ratio * σ od średniej odległości do sąsiadów
        są usuwane.
        """
        if self.points is None or len(self.points) < nb_neighbors + 1:
            return self

        tree = KDTree(self.points)
        dists, _ = tree.query(self.points, k=nb_neighbors + 1)
        mean_dists = dists[:, 1:].mean(axis=1)  # pomijamy odległość do siebie

        global_mean = np.mean(mean_dists)
        global_std = np.std(mean_dists)
        threshold = global_mean + std_ratio * global_std

        mask = mean_dists < threshold
        n_before = len(self.points)
        self.points = self.points[mask]

        if self.verbose:
            n_removed = n_before - len(self.points)
            print(
                f"🧹 Denoise: usunięto {n_removed} outlierów "
                f"({n_removed/n_before*100:.1f}%), "
                f"zostało {len(self.points)} punktów"
            )
        return self

    def normalize(self) -> "ScannerDataProcessor":
        """
        Normalizacja chmury punktów:
        1. Centrowanie (centroid → origin)
        2. Orientacja PCA (główne osie → osie XYZ) z deterministycznym znakiem
        3. Skalowanie do jednostkowej kuli
        """
        if self.points is None or len(self.points) == 0:
            return self

        pts = self.points.copy()

        # 1. Centrowanie
        centroid = pts.mean(axis=0)
        pts -= centroid

        # 2. PCA orientation z deterministycznym znakiem
        cov = np.cov(pts.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sortuj od największego eigenvalue
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Deterministyczny znak: kierunek w którym jest więcej punktów
        for i in range(3):
            proj = pts @ eigenvectors[:, i]
            if np.sum(proj > 0) < np.sum(proj < 0):
                eigenvectors[:, i] *= -1

        # Zapewnij prawoskrętny układ
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1

        pts = pts @ eigenvectors

        # 3. Skalowanie do [-1, 1]
        max_abs = np.max(np.abs(pts))
        if max_abs > 0:
            pts /= max_abs

        self.points = pts

        if self.verbose:
            print("📐 Normalizacja: centroid → origin, PCA orientation, unit sphere")
        return self

    def extract_descriptor(
        self, model_id: str = "scan", model_name: str = "Skan 3D"
    ) -> Model3DDescriptor:
        """
        Ekstrakcja deskryptora geometrycznego z chmury punktów.

        Returns:
            Model3DDescriptor z wypełnionymi cechami
        """
        if self.points is None or len(self.points) < 10:
            raise ValueError("Za mało punktów do ekstrakcji cech")

        pts = self.points
        desc = Model3DDescriptor(
            model_id=model_id,
            model_name=model_name,
            file_path=self.source_path,
            source="scan",
            num_vertices=len(pts),
        )

        # Bounding box
        bb_min = pts.min(axis=0)
        bb_max = pts.max(axis=0)
        bb = bb_max - bb_min
        desc.bounding_box = bb.tolist()
        desc.centroid = pts.mean(axis=0).tolist()

        # Convex hull → volume, surface area
        try:
            hull = ConvexHull(pts)
            desc.volume = float(hull.volume)
            desc.surface_area = float(hull.area)
            desc.num_faces = len(hull.simplices)
        except Exception:
            desc.volume = 0.0
            desc.surface_area = 0.0

        # Compactness: V / (4/3·π·r³) where r = max distance from centroid
        centroid = pts.mean(axis=0)
        dists_from_center = np.linalg.norm(pts - centroid, axis=1)
        r_max = np.max(dists_from_center)
        if r_max > 0:
            sphere_vol = (4.0 / 3.0) * np.pi * (r_max ** 3)
            desc.compactness = desc.volume / sphere_vol if sphere_vol > 0 else 0.0
        else:
            desc.compactness = 0.0

        # Sphericity: π^(1/3) * (6V)^(2/3) / A
        if desc.surface_area > 0:
            desc.sphericity = (
                (np.pi ** (1 / 3)) * ((6 * desc.volume) ** (2 / 3)) / desc.surface_area
            )
        else:
            desc.sphericity = 0.0

        # Inertia tensor eigenvalues
        pts_centered = pts - centroid
        cov = np.cov(pts_centered.T)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
        desc.inertia_eigenvalues = eigenvalues.tolist()

        # Elongation: ratio of 2nd to 1st principal axis
        if eigenvalues[0] > 0:
            desc.elongation = float(eigenvalues[1] / eigenvalues[0])
        else:
            desc.elongation = 0.0

        # D2 histogram: distribution of pairwise distances
        desc.d2_histogram = self._compute_d2_histogram(pts, n_bins=64)

        # A3 histogram: distribution of angles (triples of points)
        desc.a3_histogram = self._compute_a3_histogram(pts, n_bins=64)

        # Geometric hash
        desc.model_hash = self._compute_geometry_hash(pts)

        return desc

    @staticmethod
    def _compute_d2_histogram(
        pts: np.ndarray, n_bins: int = 64, n_samples: int = 50000
    ) -> List[float]:
        """
        Histogram D2: rozkład odległości między parami losowo wybranych punktów.
        (Shape Distribution — Osada et al., 2002)
        Używa deterministycznego seed dla powtarzalności.
        """
        n = len(pts)
        if n < 2:
            return [0.0] * n_bins

        rng = np.random.RandomState(12345)  # Deterministyczny seed
        n_pairs = min(n_samples, n * (n - 1) // 2)
        idx1 = rng.randint(0, n, size=n_pairs)
        idx2 = rng.randint(0, n, size=n_pairs)
        # Unikaj par z samym sobą
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]

        distances = np.linalg.norm(pts[idx1] - pts[idx2], axis=1)

        # Zakres [0, max_dist] zamiast auto — stabilne biny
        max_dist = np.max(distances) if len(distances) > 0 else 1.0
        hist, _ = np.histogram(distances, bins=n_bins, range=(0, max_dist), density=True)
        return hist.tolist()

    @staticmethod
    def _compute_a3_histogram(
        pts: np.ndarray, n_bins: int = 64, n_samples: int = 30000
    ) -> List[float]:
        """
        Histogram A3: rozkład kątów między trójkami losowo wybranych punktów.
        (Shape Distribution — Osada et al., 2002)
        Używa deterministycznego seed dla powtarzalności.
        """
        n = len(pts)
        if n < 3:
            return [0.0] * n_bins

        rng = np.random.RandomState(67890)  # Deterministyczny seed
        n_triples = min(n_samples, n * (n - 1) * (n - 2) // 6)
        idx1 = rng.randint(0, n, size=n_triples)
        idx2 = rng.randint(0, n, size=n_triples)
        idx3 = rng.randint(0, n, size=n_triples)

        # Unikaj powtórzeń
        mask = (idx1 != idx2) & (idx2 != idx3) & (idx1 != idx3)
        idx1, idx2, idx3 = idx1[mask], idx2[mask], idx3[mask]

        if len(idx1) == 0:
            return [0.0] * n_bins

        v1 = pts[idx1] - pts[idx2]
        v2 = pts[idx3] - pts[idx2]

        # Normalizacja
        n1 = np.linalg.norm(v1, axis=1, keepdims=True)
        n2 = np.linalg.norm(v2, axis=1, keepdims=True)
        valid = (n1.flatten() > 1e-10) & (n2.flatten() > 1e-10)
        v1, v2, n1, n2 = v1[valid], v2[valid], n1[valid], n2[valid]

        if len(v1) == 0:
            return [0.0] * n_bins

        cos_angles = np.sum((v1 / n1) * (v2 / n2), axis=1)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)

        hist, _ = np.histogram(angles, bins=n_bins, range=(0, np.pi), density=True)
        return hist.tolist()

    @staticmethod
    def _compute_geometry_hash(pts: np.ndarray) -> str:
        """SHA-256 hash geometrii (fingerprint do dokładnej identyfikacji)."""
        # Sortuj i zaokrąglij punkty do 4 miejsc po przecinku
        sorted_pts = pts[np.lexsort(pts.T)]
        rounded = np.round(sorted_pts, 4)
        data = rounded.tobytes()
        return hashlib.sha256(data).hexdigest()


# ============================================================================
# 4. EKSTRAKCJA CECH Z PLIKÓW 3D (mesh)
# ============================================================================


def extract_descriptor_from_file(
    filepath: str,
    model_id: str = "",
    model_name: str = "",
    category: str = "",
    material: str = "",
    tags: Optional[List[str]] = None,
    extra_points: Optional[np.ndarray] = None,
) -> Model3DDescriptor:
    """
    Ekstrakcja deskryptora geometrycznego bezpośrednio z pliku 3D (STL/OBJ/PLY).

    Ładuje mesh, próbkuje punkty i oblicza cechy.
    Dodatkowo wyciąga informacje z samego mesha (faces, volume, etc.).

    Args:
        filepath: Ścieżka do pliku 3D
        model_id: ID modelu (domyślnie = nazwa pliku)
        model_name: Czytelna nazwa (domyślnie = nazwa pliku)
        category: Kategoria części
        material: Materiał
        tags: Tagi
        extra_points: Dodatkowe punkty (np. wewnętrzne powierzchnie otworów)

    Returns:
        Model3DDescriptor
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("Wymagany pakiet: pip install trimesh")

    path = Path(filepath)
    if not model_id:
        model_id = path.stem
    if not model_name:
        model_name = path.stem

    mesh = trimesh.load(str(filepath), force="mesh")

    # Próbkuj punkty
    n_samples = min(10000, max(5000, len(mesh.vertices) * 2))
    points = np.array(mesh.sample(n_samples), dtype=np.float64)

    # Merge extra points if provided (e.g. inner surfaces of hollow shapes)
    if extra_points is not None and len(extra_points) > 0:
        points = np.vstack([points, extra_points])

    processor = ScannerDataProcessor(verbose=False)
    processor.points = points
    processor.source_path = str(filepath)

    desc = processor.extract_descriptor(model_id=model_id, model_name=model_name)

    # Nadpisz danymi z mesha (dokładniejsze niż z convex hull)
    if mesh.is_watertight:
        desc.volume = float(mesh.volume)
    desc.surface_area = float(mesh.area)
    desc.num_vertices = len(mesh.vertices)
    desc.num_faces = len(mesh.faces)
    desc.file_path = str(filepath)
    desc.category = category
    desc.material = material
    desc.tags = tags or []
    desc.source = "cad"

    # Bounding box z mesha
    bb = mesh.bounding_box.extents
    desc.bounding_box = bb.tolist()

    # Centroid z mesha
    desc.centroid = mesh.centroid.tolist()

    # Inertia z mesha
    try:
        inertia = mesh.moment_inertia
        eigenvalues = np.sort(np.linalg.eigvalsh(inertia))[::-1]
        desc.inertia_eigenvalues = eigenvalues.tolist()
        if eigenvalues[0] > 0:
            desc.elongation = float(eigenvalues[1] / eigenvalues[0])
    except Exception:
        pass

    # Hash
    desc.model_hash = hashlib.sha256(
        np.round(mesh.vertices, 4).tobytes()
    ).hexdigest()

    return desc


# ============================================================================
# 5. ALGORYTM PORÓWNYWANIA — ICP + HAUSDORFF + DESKRYPTORY
# ============================================================================


class Model3DComparator:
    """
    Algorytm porównywania danych ze skanera z modelami 3D z bazy danych.

    Pipeline identyfikacji:
    ┌─────────────────────────────────────────────────────┐
    │  1. COARSE MATCHING (szybkie)                       │
    │     Porównanie deskryptorów geometrycznych           │
    │     → Top-K kandydatów                               │
    ├─────────────────────────────────────────────────────┤
    │  2. FINE MATCHING (dokładne, Top-K)                  │
    │     a) ICP alignment (Iterative Closest Point)       │
    │     b) Hausdorff distance                            │
    │     c) Point-to-point RMSE                           │
    ├─────────────────────────────────────────────────────┤
    │  3. SCORING & RANKING                                │
    │     Łączna ocena: descriptor_sim + ICP + Hausdorff   │
    │     → Zidentyfikowany model                          │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self, database: Model3DDatabase, verbose: bool = True):
        self.database = database
        self.verbose = verbose

    def identify(
        self,
        scan_points: np.ndarray,
        top_k: int = 10,
        refine_top: int = 8,
        category_filter: Optional[str] = None,
        icp_iterations: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Pełny pipeline identyfikacji modelu z biblioteki na podstawie skanu.

        Args:
            scan_points: Chmura punktów ze skanera (N, 3)
            top_k: Ile kandydatów z wstępnego wyszukiwania
            refine_top: Ile najlepszych kandydatów poddać ICP
            category_filter: Filtr kategorii
            icp_iterations: Liczba iteracji ICP

        Returns:
            Lista wyników: [
                {
                    "model_id": str,
                    "model_name": str,
                    "descriptor_similarity": float,
                    "icp_fitness": float,
                    "icp_rmse": float,
                    "hausdorff_distance": float,
                    "final_score": float,
                    "descriptor": Model3DDescriptor,
                },
                ...
            ]
        """
        t_start = time.time()

        if self.verbose:
            print("=" * 70)
            print("🔍 IDENTYFIKACJA MODELU 3D Z BIBLIOTEKI")
            print("=" * 70)

        # ----- Krok 1: Przetwarzanie danych skanera -----
        if self.verbose:
            print(f"\n📥 Dane wejściowe: {len(scan_points)} punktów")

        processor = ScannerDataProcessor(verbose=self.verbose)
        processor.points = scan_points.copy()
        processor.downsample(target_points=5000)
        processor.denoise(nb_neighbors=15, std_ratio=2.0)
        processor.normalize()

        scan_descriptor = processor.extract_descriptor(
            model_id="query_scan", model_name="Zapytanie skanera"
        )

        # ----- Krok 2: Coarse matching — deskryptory -----
        if self.verbose:
            print(f"\n🔎 Faza 1: Wyszukiwanie wstępne (deskryptory → Top-{top_k})")

        coarse_results = self.database.search(
            scan_descriptor,
            top_k=top_k,
            category_filter=category_filter,
        )

        if not coarse_results:
            if self.verbose:
                print("❌ Brak modeli w bazie danych!")
            return []

        if self.verbose:
            for i, (desc, score) in enumerate(coarse_results, 1):
                print(f"   #{i}: {desc.model_name} [{desc.model_id}] "
                      f"— similarity: {score:.4f}")

        # ----- Krok 3: Fine matching — ICP + Hausdorff -----
        if self.verbose:
            print(f"\n🔬 Faza 2: Dopasowanie precyzyjne (ICP + Hausdorff → Top-{refine_top})")

        candidates = coarse_results[:refine_top]
        refined_results = []

        scan_pts_normalized = processor.points

        for desc, coarse_score in candidates:
            # Załaduj punkty modelu referencyjnego
            ref_pts = self._get_reference_points(desc)

            if ref_pts is None:
                # Jeśli brak pliku — użyj samego deskryptora
                refined_results.append({
                    "model_id": desc.model_id,
                    "model_name": desc.model_name,
                    "category": desc.category,
                    "material": desc.material,
                    "descriptor_similarity": coarse_score,
                    "icp_fitness": -1.0,
                    "icp_rmse": -1.0,
                    "hausdorff_distance": -1.0,
                    "final_score": coarse_score,
                    "descriptor": desc,
                })
                continue

            # Normalizuj punkty referencyjne
            ref_processor = ScannerDataProcessor(verbose=False)
            ref_processor.points = ref_pts
            ref_processor.downsample(target_points=5000)
            ref_processor.normalize()
            ref_pts_norm = ref_processor.points

            # ICP alignment with multiple initial orientations for robustness
            icp_result = self._multi_orientation_icp(
                scan_pts_normalized, ref_pts_norm,
                max_iterations=icp_iterations,
            )

            # Hausdorff + Chamfer distance
            hd = self._hausdorff_distance(
                icp_result["transformed_points"], ref_pts_norm
            )
            cd = self._chamfer_distance(
                icp_result["transformed_points"], ref_pts_norm
            )

            # Final score = weighted combination
            # Deskryptor jest bardziej wiarygodny dla klasyfikacji kształtu,
            # ICP/Hausdorff/Chamfer weryfikują dopasowanie geometrii
            desc_weight = 0.50
            icp_weight = 0.20
            hd_weight = 0.10
            cd_weight = 0.20

            icp_score = max(0, 1.0 - icp_result["rmse"] * 3)
            hd_score = max(0, 1.0 - hd * 1.5)
            cd_score = max(0, 1.0 - cd * 5)

            final_score = (
                desc_weight * coarse_score
                + icp_weight * icp_score
                + hd_weight * hd_score
                + cd_weight * cd_score
            )

            refined_results.append({
                "model_id": desc.model_id,
                "model_name": desc.model_name,
                "category": desc.category,
                "material": desc.material,
                "descriptor_similarity": coarse_score,
                "icp_fitness": icp_result["fitness"],
                "icp_rmse": icp_result["rmse"],
                "hausdorff_distance": hd,
                "chamfer_distance": cd,
                "final_score": final_score,
                "descriptor": desc,
            })

            if self.verbose:
                print(
                    f"   {desc.model_name}: "
                    f"descriptor={coarse_score:.4f}, "
                    f"ICP_rmse={icp_result['rmse']:.6f}, "
                    f"Hausdorff={hd:.6f}, "
                    f"FINAL={final_score:.4f}"
                )

        # Dodaj pozostałych kandydatów (bez ICP)
        for desc, coarse_score in coarse_results[refine_top:]:
            refined_results.append({
                "model_id": desc.model_id,
                "model_name": desc.model_name,
                "category": desc.category,
                "material": desc.material,
                "descriptor_similarity": coarse_score,
                "icp_fitness": -1.0,
                "icp_rmse": -1.0,
                "hausdorff_distance": -1.0,
                "final_score": coarse_score * 0.4,
                "descriptor": desc,
            })

        # Sortuj po final_score
        refined_results.sort(key=lambda x: x["final_score"], reverse=True)

        t_elapsed = (time.time() - t_start) * 1000

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"✅ WYNIK IDENTYFIKACJI (czas: {t_elapsed:.0f} ms)")
            print(f"{'='*70}")
            if refined_results:
                best = refined_results[0]
                print(f"   🏆 Najlepsze dopasowanie: {best['model_name']} "
                      f"[{best['model_id']}]")
                print(f"      Kategoria: {best['category']}")
                print(f"      Materiał: {best['material']}")
                print(f"      Final Score: {best['final_score']:.4f}")
                if best["icp_rmse"] >= 0:
                    print(f"      ICP RMSE: {best['icp_rmse']:.6f}")
                    print(f"      Hausdorff: {best['hausdorff_distance']:.6f}")
            print(f"{'='*70}")

        # Log do bazy
        if refined_results:
            best = refined_results[0]
            self.database.log_comparison(
                scan_source=processor.source_path or "programmatic",
                matched_model_id=best["model_id"],
                similarity_score=best["final_score"],
                method="full_pipeline",
                hausdorff_distance=best.get("hausdorff_distance", -1),
                icp_fitness=best.get("icp_fitness", -1),
                icp_rmse=best.get("icp_rmse", -1),
                num_candidates=len(coarse_results),
                processing_time_ms=t_elapsed,
            )

        return refined_results

    def quick_identify(
        self,
        scan_points: np.ndarray,
        top_k: int = 5,
        category_filter: Optional[str] = None,
    ) -> List[Tuple[Model3DDescriptor, float]]:
        """
        Szybka identyfikacja — tylko na podstawie deskryptorów (bez ICP).
        Idealne do wstępnego przesiewu.
        """
        processor = ScannerDataProcessor(verbose=False)
        processor.points = scan_points.copy()
        processor.downsample(target_points=3000)
        processor.normalize()

        descriptor = processor.extract_descriptor(
            model_id="quick_scan", model_name="Quick scan"
        )

        return self.database.search(
            descriptor, top_k=top_k,
            category_filter=category_filter,
        )

    def compare_two_meshes(
        self,
        points_a: np.ndarray,
        points_b: np.ndarray,
        icp_iterations: int = 50,
    ) -> Dict[str, Any]:
        """
        Porównanie dwóch chmur punktów (np. skan vs model CAD).

        Returns:
            {
                "descriptor_similarity": float,
                "icp_fitness": float,
                "icp_rmse": float,
                "hausdorff_distance": float,
                "hausdorff_distance_reverse": float,
                "chamfer_distance": float,
                "final_score": float,
            }
        """
        proc_a = ScannerDataProcessor(verbose=False)
        proc_a.points = points_a.copy()
        proc_a.downsample(5000)
        proc_a.normalize()

        proc_b = ScannerDataProcessor(verbose=False)
        proc_b.points = points_b.copy()
        proc_b.downsample(5000)
        proc_b.normalize()

        desc_a = proc_a.extract_descriptor("mesh_a", "Mesh A")
        desc_b = proc_b.extract_descriptor("mesh_b", "Mesh B")

        fv_a = desc_a.to_global_feature_vector()
        fv_b = desc_b.to_global_feature_vector()
        desc_sim = Model3DDatabase._cosine_similarity(fv_a, fv_b)

        # ICP
        icp_result = self._icp(proc_a.points, proc_b.points, icp_iterations)

        # Hausdorff (oba kierunki)
        hd_ab = self._hausdorff_distance(icp_result["transformed_points"], proc_b.points)
        hd_ba = self._hausdorff_distance(proc_b.points, icp_result["transformed_points"])

        # Chamfer distance
        chamfer = self._chamfer_distance(icp_result["transformed_points"], proc_b.points)

        # Final score
        icp_score = max(0, 1.0 - icp_result["rmse"] * 5)
        hd_score = max(0, 1.0 - max(hd_ab, hd_ba) * 2)
        final_score = 0.4 * desc_sim + 0.35 * icp_score + 0.25 * hd_score

        return {
            "descriptor_similarity": desc_sim,
            "icp_fitness": icp_result["fitness"],
            "icp_rmse": icp_result["rmse"],
            "hausdorff_distance": hd_ab,
            "hausdorff_distance_reverse": hd_ba,
            "chamfer_distance": chamfer,
            "final_score": final_score,
        }

    # ---- Algorytmy niskopoziomowe ----

    @staticmethod
    def _multi_orientation_icp(
        source: np.ndarray,
        target: np.ndarray,
        max_iterations: int = 50,
        n_orientations: int = 4,
    ) -> Dict[str, Any]:
        """
        Multi-start ICP — próbuje kilka początkowych orientacji
        i wybiera najlepszą (najniższe RMSE).

        Rozwiązuje problem ambiguity PCA (flip/rotate).
        4 orientacje: identity + 3 obroty 180° wokół osi głównych.
        """
        rotations = [
            np.eye(3),                                    # identity
            np.diag([-1, -1,  1]),                        # 180° wokół Z
            np.diag([-1,  1, -1]),                        # 180° wokół Y
            np.diag([ 1, -1, -1]),                        # 180° wokół X
        ]

        best_result = None
        best_rmse = float("inf")

        for R_init in rotations[:n_orientations]:
            src_rotated = (R_init @ source.T).T
            result = Model3DComparator._icp(
                src_rotated, target, max_iterations=max_iterations
            )
            if result["rmse"] < best_rmse:
                best_rmse = result["rmse"]
                best_result = result

        return best_result

    # ---- Single ICP ----

    @staticmethod
    def _icp(
        source: np.ndarray,
        target: np.ndarray,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
    ) -> Dict[str, Any]:
        """
        Iterative Closest Point (ICP) — dopasowanie dwóch chmur punktów.

        Implementacja point-to-point ICP z SVD.

        Args:
            source: Chmura punktów źródłowa (N, 3)
            target: Chmura punktów docelowa (M, 3)
            max_iterations: Maksymalna liczba iteracji
            tolerance: Próg zbieżności (zmiana RMSE)

        Returns:
            {
                "transformation": (4, 4) macierz transformacji,
                "transformed_points": (N, 3) dopasowane punkty,
                "rmse": float — Root Mean Square Error,
                "fitness": float — odsetek punktów bliżej niż 2*RMSE,
                "iterations": int,
            }
        """
        src = source.copy()
        prev_rmse = float("inf")
        transformation = np.eye(4)
        target_tree = KDTree(target)

        for iteration in range(max_iterations):
            # Znajdź najbliższe punkty
            dists, indices = target_tree.query(src)

            closest = target[indices]

            # Centrowanie
            src_centroid = src.mean(axis=0)
            tgt_centroid = closest.mean(axis=0)

            src_centered = src - src_centroid
            tgt_centered = closest - tgt_centroid

            # SVD do obliczenia rotacji
            H = src_centered.T @ tgt_centered
            U, S, Vt = svd(H)
            R = Vt.T @ U.T

            # Korekta refleksji
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            # Translacja
            t = tgt_centroid - R @ src_centroid

            # Zastosuj transformację
            src = (R @ src.T).T + t

            # Aktualizuj macierz transformacji
            T_step = np.eye(4)
            T_step[:3, :3] = R
            T_step[:3, 3] = t
            transformation = T_step @ transformation

            # RMSE
            rmse = np.sqrt(np.mean(dists ** 2))

            if abs(prev_rmse - rmse) < tolerance:
                break
            prev_rmse = rmse

        # Fitness: odsetek punktów bliżej niż threshold
        final_dists, _ = target_tree.query(src)
        threshold = 2.0 * np.mean(final_dists)
        fitness = float(np.sum(final_dists < threshold) / len(final_dists))

        return {
            "transformation": transformation,
            "transformed_points": src,
            "rmse": float(np.sqrt(np.mean(final_dists ** 2))),
            "fitness": fitness,
            "iterations": iteration + 1,
        }

    @staticmethod
    def _hausdorff_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
        """
        Hausdorff distance — maksymalne odstępstwo między chmurami.
        Używamy directed_hausdorff z scipy.
        """
        hd, _, _ = directed_hausdorff(pts_a, pts_b)
        return float(hd)

    @staticmethod
    def _chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
        """
        Chamfer distance — średnie odległości nearest-neighbor w obu kierunkach.
        """
        tree_a = KDTree(pts_a)
        tree_b = KDTree(pts_b)

        dists_a_to_b, _ = tree_b.query(pts_a)
        dists_b_to_a, _ = tree_a.query(pts_b)

        return float(np.mean(dists_a_to_b ** 2) + np.mean(dists_b_to_a ** 2))

    def _get_reference_points(self, desc: Model3DDescriptor) -> Optional[np.ndarray]:
        """Załaduj punkty referencyjne z pliku modelu."""
        if not desc.file_path or not os.path.exists(desc.file_path):
            return None
        try:
            proc = ScannerDataProcessor(verbose=False)
            proc.load(desc.file_path)
            return proc.points
        except Exception:
            return None


# ============================================================================
# 6. GENERACJA DANYCH DEMONSTRACYJNYCH
# ============================================================================


def generate_demo_3d_models(
    output_dir: str = "demo_3d_models",
    db_path: str = "models_3d.db",
) -> Model3DDatabase:
    """
    Generuje zestaw demonstracyjnych modeli 3D (parametrycznych)
    i zapisuje je do bazy danych.

    Modele generowane programowo (trimesh primitives):
        - Śruba M8 (cylinder + head)
        - Śruba M6 (cylinder + head, mniejsza)
        - Nakrętka M8 (annular hexagonal prism)
        - Łożysko (torus + cylinder)
        - Wał (long cylinder)
        - Kołek (short cylinder)
        - Tuleja (hollow cylinder)
        - Sprężyna (helix approximation)
        - Płyta (box)
        - Kula (sphere)

    Returns:
        Model3DDatabase z załadowanymi modelami
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh wymagany do generacji modeli demo")

    os.makedirs(output_dir, exist_ok=True)
    db = Model3DDatabase(db_path)

    def _sample_hollow_cylinder(r_outer, r_inner, height, sections=64, n_pts=5000):
        """Generate point cloud for a hollow cylinder (ring/annulus shape)."""
        pts = []
        n_ring = n_pts // 3
        n_inner = n_pts // 3
        n_top = n_pts - n_ring - n_inner

        # Inner cylinder surface
        for _ in range(n_inner):
            theta = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(-height / 2, height / 2)
            pts.append([r_inner * np.cos(theta), r_inner * np.sin(theta), z])

        # Top/bottom ring faces
        for _ in range(n_top):
            r = np.random.uniform(r_inner, r_outer)
            theta = np.random.uniform(0, 2 * np.pi)
            z = (height / 2) * np.random.choice([-1, 1])
            pts.append([r * np.cos(theta), r * np.sin(theta), z])

        # Outer surface (already covered by mesh sampling)
        return np.array(pts, dtype=np.float64)

    models = []

    # ---- 1. Śruba M8x20 ----
    head = trimesh.creation.cylinder(radius=6.5, height=5.3, sections=6)
    head.apply_translation([0, 0, 2.65])
    shaft = trimesh.creation.cylinder(radius=4.0, height=20.0, sections=32)
    shaft.apply_translation([0, 0, -10.0 + 5.3])
    bolt_m8 = trimesh.util.concatenate([head, shaft])
    models.append({
        "mesh": bolt_m8,
        "model_id": "BOLT-M8x20",
        "name": "Śruba sześciokątna M8x20",
        "category": "fasteners",
        "material": "Stal nierdzewna A2-70",
        "tags": ["śruba", "M8", "hex", "metryczna"],
    })

    # ---- 2. Śruba M6x16 ----
    head6 = trimesh.creation.cylinder(radius=5.0, height=4.0, sections=6)
    head6.apply_translation([0, 0, 2.0])
    shaft6 = trimesh.creation.cylinder(radius=3.0, height=16.0, sections=32)
    shaft6.apply_translation([0, 0, -8.0 + 4.0])
    bolt_m6 = trimesh.util.concatenate([head6, shaft6])
    models.append({
        "mesh": bolt_m6,
        "model_id": "BOLT-M6x16",
        "name": "Śruba sześciokątna M6x16",
        "category": "fasteners",
        "material": "Stal nierdzewna A2-70",
        "tags": ["śruba", "M6", "hex", "metryczna"],
    })

    # ---- 3. Nakrętka M8 (hexagonal prism — ring shape via annulus sampling) ----
    nut_m8 = trimesh.creation.cylinder(radius=6.5, height=6.5, sections=6)
    # Add inner hole as additional geometry (creating annular shape)
    nut_inner_pts = _sample_hollow_cylinder(r_outer=6.5, r_inner=4.0, height=6.5, sections=6, n_pts=5000)
    models.append({
        "mesh": nut_m8,
        "model_id": "NUT-M8",
        "name": "Nakrętka sześciokątna M8",
        "category": "fasteners",
        "material": "Stal nierdzewna A2-70",
        "tags": ["nakrętka", "M8", "hex"],
        "extra_points": nut_inner_pts,
    })

    # ---- 4. Łożysko 6205 (large ring shape) ----
    bearing = trimesh.creation.cylinder(radius=26.0, height=15.0, sections=64)
    bearing_inner_pts = _sample_hollow_cylinder(r_outer=26.0, r_inner=12.5, height=15.0, sections=64, n_pts=5000)
    models.append({
        "mesh": bearing,
        "model_id": "BRG-6205",
        "name": "Łożysko kulkowe 6205-2RS",
        "category": "bearings",
        "material": "Stal łożyskowa",
        "tags": ["łożysko", "kulkowe", "6205"],
        "extra_points": bearing_inner_pts,
    })

    # ---- 5. Wał 12mm x 300mm ----
    shaft_model = trimesh.creation.cylinder(radius=6.0, height=300.0, sections=64)
    models.append({
        "mesh": shaft_model,
        "model_id": "SHAFT-12x300",
        "name": "Wał chromowany 12mm x 300mm",
        "category": "shafts",
        "material": "Stal chromowana",
        "tags": ["wał", "chromowany", "12mm"],
    })

    # ---- 6. Kołek cylindryczny 6x30 ----
    pin_model = trimesh.creation.cylinder(radius=3.0, height=30.0, sections=32)
    models.append({
        "mesh": pin_model,
        "model_id": "PIN-6x30",
        "name": "Kołek cylindryczny 6x30",
        "category": "fasteners",
        "material": "Stal hartowana",
        "tags": ["kołek", "cylindryczny", "6mm"],
    })

    # ---- 7. Tuleja dystansowa 10x15x20 ----
    sleeve = trimesh.creation.cylinder(radius=7.5, height=20.0, sections=64)
    sleeve_inner_pts = _sample_hollow_cylinder(r_outer=7.5, r_inner=5.0, height=20.0, sections=64, n_pts=5000)
    models.append({
        "mesh": sleeve,
        "model_id": "SLEEVE-10x15x20",
        "name": "Tuleja dystansowa 10/15x20",
        "category": "bushings",
        "material": "Brąz",
        "tags": ["tuleja", "dystansowa", "brązowa"],
        "extra_points": sleeve_inner_pts,
    })

    # ---- 8. Podkładka M8 ----
    washer = trimesh.creation.cylinder(radius=8.0, height=1.6, sections=64)
    washer_inner_pts = _sample_hollow_cylinder(r_outer=8.0, r_inner=4.3, height=1.6, sections=64, n_pts=5000)
    models.append({
        "mesh": washer,
        "model_id": "WASHER-M8",
        "name": "Podkładka płaska M8",
        "category": "fasteners",
        "material": "Stal ocynkowana",
        "tags": ["podkładka", "M8", "płaska"],
        "extra_points": washer_inner_pts,
    })

    # ---- 9. Płyta 100x100x10 ----
    plate = trimesh.creation.box(extents=[100.0, 100.0, 10.0])
    models.append({
        "mesh": plate,
        "model_id": "PLATE-100x100x10",
        "name": "Płyta stalowa 100x100x10",
        "category": "plates",
        "material": "Stal konstrukcyjna S235",
        "tags": ["płyta", "prostokątna", "stalowa"],
    })

    # ---- 10. Kula ø30 ----
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=15.0)
    models.append({
        "mesh": sphere,
        "model_id": "SPHERE-30",
        "name": "Kula stalowa ø30",
        "category": "other",
        "material": "Stal łożyskowa",
        "tags": ["kula", "stalowa", "precyzyjna"],
    })

    # Zapisz do plików STL i do bazy
    print(f"📦 Generowanie {len(models)} modeli demonstracyjnych...")
    for m in models:
        stl_path = os.path.join(output_dir, f"{m['model_id']}.stl")
        m["mesh"].export(stl_path)

        # Use extra_points if available (for hollow shapes)
        extra_pts = m.get("extra_points", None)

        desc = extract_descriptor_from_file(
            stl_path,
            model_id=m["model_id"],
            model_name=m["name"],
            category=m["category"],
            material=m["material"],
            tags=m["tags"],
            extra_points=extra_pts,
        )
        db.add_model(desc)

        print(f"   ✅ {m['model_id']}: {m['name']} → {stl_path}")

    print(f"\n📊 Modeli w bazie: {db.count()}")
    print(f"💾 Baza danych: {db_path}")
    print(f"📁 Pliki STL: {output_dir}/")

    return db


def generate_demo_scan(
    model_id: str,
    models_dir: str = "demo_3d_models",
    noise_level: float = 0.02,
    missing_ratio: float = 0.1,
    n_points: int = 8000,
) -> np.ndarray:
    """
    Generuje symulowane dane skanera 3D na podstawie modelu referencyjnego.

    Symulacja obejmuje:
        - Próbkowanie punktów z powierzchni modelu
        - Dodanie szumu Gaussa (symulacja niedokładności skanera)
        - Losowe usunięcie punktów (symulacja okluzji/braków)
        - Losowy obrót i przesunięcie (symulacja dowolnej orientacji)

    Args:
        model_id: ID modelu do zasymulowania skanu
        models_dir: Katalog z plikami STL
        noise_level: Poziom szumu (std dev) jako ułamek bounding box
        missing_ratio: Ułamek brakujących punktów (0-1)
        n_points: Liczba punktów w skanie

    Returns:
        np.ndarray (N, 3) — symulowana chmura punktów
    """
    stl_path = os.path.join(models_dir, f"{model_id}.stl")
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"Nie znaleziono: {stl_path}")

    mesh = trimesh.load(stl_path)
    points = np.array(mesh.sample(n_points), dtype=np.float64)

    # For hollow models, add inner surface points based on known dimensions
    _hollow_models = {
        "NUT-M8": (6.5, 4.0, 6.5, 6),
        "BRG-6205": (26.0, 12.5, 15.0, 64),
        "SLEEVE-10x15x20": (7.5, 5.0, 20.0, 64),
        "WASHER-M8": (8.0, 4.3, 1.6, 64),
    }
    if model_id in _hollow_models:
        r_out, r_in, h, sec = _hollow_models[model_id]
        n_inner = n_points // 3
        inner_pts = []
        for _ in range(n_inner):
            theta = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(-h / 2, h / 2)
            r = r_in if np.random.random() < 0.5 else np.random.uniform(r_in, r_out)
            face = np.random.random()
            if face < 0.33:
                inner_pts.append([r_in * np.cos(theta), r_in * np.sin(theta), z])
            elif face < 0.66:
                inner_pts.append([r * np.cos(theta), r * np.sin(theta), h / 2 * np.random.choice([-1, 1])])
            else:
                inner_pts.append([r_in * np.cos(theta), r_in * np.sin(theta), z])
        points = np.vstack([points, np.array(inner_pts, dtype=np.float64)])

    # Bounding box
    bb = np.max(points, axis=0) - np.min(points, axis=0)
    bb_diag = np.linalg.norm(bb)

    # 1. Dodaj szum
    noise = np.random.normal(0, noise_level * bb_diag, size=points.shape)
    points += noise

    # 2. Usuń losowe punkty (okluzja)
    n_keep = int(len(points) * (1 - missing_ratio))
    indices = np.random.choice(len(points), size=n_keep, replace=False)
    points = points[indices]

    # 3. Losowy obrót
    angle_x = np.random.uniform(0, 2 * np.pi)
    angle_y = np.random.uniform(0, 2 * np.pi)
    angle_z = np.random.uniform(0, 2 * np.pi)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)],
    ])
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)],
    ])
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1],
    ])

    R = Rz @ Ry @ Rx
    points = (R @ points.T).T

    # 4. Losowe przesunięcie
    translation = np.random.uniform(-10, 10, size=3)
    points += translation

    return points


# ============================================================================
# 7. NARZĘDZIA WIZUALIZACJI
# ============================================================================


def plot_identification_results(
    results: List[Dict[str, Any]],
    scan_points: Optional[np.ndarray] = None,
    title: str = "Identyfikacja modelu 3D",
):
    """
    Wizualizacja wyników identyfikacji.

    Args:
        results: Lista wyników z Model3DComparator.identify()
        scan_points: Opcjonalnie — chmura punktów skanera do wyświetlenia
        title: Tytuł wykresu
    """
    import matplotlib.pyplot as plt

    n_results = min(len(results), 5)
    fig_height = 4 + n_results * 0.8

    if scan_points is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, fig_height))

        # Wykres 3D skanera
        ax3d = fig.add_subplot(121, projection="3d")
        pts = scan_points
        if len(pts) > 3000:
            idx = np.random.choice(len(pts), 3000, replace=False)
            pts = pts[idx]
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, alpha=0.6, c="steelblue")
        ax3d.set_title("Chmura punktów skanera")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")

        # Wykres wyników
        ax_bar = axes[1]
        axes[0].set_visible(False)
    else:
        fig, ax_bar = plt.subplots(1, 1, figsize=(10, fig_height))

    # Bar chart wyników
    labels = [f"{r['model_name']}\n[{r['model_id']}]" for r in results[:n_results]]
    scores = [r["final_score"] for r in results[:n_results]]
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(n_results)]

    bars = ax_bar.barh(range(n_results), scores, color=colors, edgecolor="white", height=0.6)
    ax_bar.set_yticks(range(n_results))
    ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.set_xlabel("Score")
    ax_bar.set_title(title)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.1)

    # Adnotacje
    for i, (bar, score) in enumerate(zip(bars, scores)):
        r = results[i]
        extra = ""
        if r.get("icp_rmse", -1) >= 0:
            extra = f" | ICP={r['icp_rmse']:.4f}"
        ax_bar.text(
            score + 0.02, i, f"{score:.3f}{extra}",
            va="center", fontsize=9
        )

    plt.tight_layout()
    plt.show()


def plot_point_clouds_comparison(
    scan_pts: np.ndarray,
    ref_pts: np.ndarray,
    title: str = "Porównanie chmur punktów",
    max_display: int = 3000,
):
    """Wizualizacja dwóch chmur punktów w 3D."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 5))

    # Downsampling do wyświetlenia
    s_pts = scan_pts
    r_pts = ref_pts
    if len(s_pts) > max_display:
        s_pts = s_pts[np.random.choice(len(s_pts), max_display, replace=False)]
    if len(r_pts) > max_display:
        r_pts = r_pts[np.random.choice(len(r_pts), max_display, replace=False)]

    # Widok 1: Osobno
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(s_pts[:, 0], s_pts[:, 1], s_pts[:, 2], s=0.5, alpha=0.5, c="blue", label="Skan")
    ax1.set_title("Skan 3D")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Widok 2: Nałożone
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(s_pts[:, 0], s_pts[:, 1], s_pts[:, 2], s=0.5, alpha=0.4, c="blue", label="Skan")
    ax2.scatter(r_pts[:, 0], r_pts[:, 1], r_pts[:, 2], s=0.5, alpha=0.4, c="red", label="Model ref.")
    ax2.set_title(title)
    ax2.legend(markerscale=10)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


# ============================================================================
# 8. RAPORT IDENTYFIKACJI
# ============================================================================


def generate_identification_report(
    results: List[Dict[str, Any]],
    scan_source: str = "Skan 3D",
) -> str:
    """
    Generuje czytelny raport tekstowy z wyników identyfikacji.

    Args:
        results: Wyniki z Model3DComparator.identify()
        scan_source: Opis źródła skanu

    Returns:
        Sformatowany raport (str)
    """
    report = []
    report.append("=" * 70)
    report.append("📋 RAPORT IDENTYFIKACJI MODELU 3D")
    report.append(f"   Źródło danych: {scan_source}")
    report.append(f"   Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    report.append("")

    if not results:
        report.append("❌ Nie znaleziono pasujących modeli w bibliotece.")
        return "\n".join(report)

    best = results[0]
    report.append("🏆 NAJLEPSZE DOPASOWANIE:")
    report.append(f"   Model:     {best['model_name']} [{best['model_id']}]")
    report.append(f"   Kategoria: {best.get('category', 'N/A')}")
    report.append(f"   Materiał:  {best.get('material', 'N/A')}")
    report.append(f"   Score:     {best['final_score']:.4f} "
                  f"({'doskonały' if best['final_score'] > 0.9 else 'bardzo dobry' if best['final_score'] > 0.7 else 'dobry' if best['final_score'] > 0.5 else 'słaby'})")

    if best.get("icp_rmse", -1) >= 0:
        report.append(f"   ICP RMSE:  {best['icp_rmse']:.6f}")
        report.append(f"   Hausdorff: {best['hausdorff_distance']:.6f}")
        report.append(f"   ICP Fit:   {best['icp_fitness']:.4f}")

    report.append("")
    report.append("─" * 70)
    report.append("📊 RANKING KANDYDATÓW:")
    report.append("─" * 70)

    for i, r in enumerate(results, 1):
        star = " ⭐" if i == 1 else ""
        report.append(
            f"  #{i}: {r['model_name']} [{r['model_id']}] "
            f"— score: {r['final_score']:.4f}{star}"
        )
        if r.get("icp_rmse", -1) >= 0:
            report.append(
                f"       ICP_rmse={r['icp_rmse']:.6f}, "
                f"Hausdorff={r['hausdorff_distance']:.6f}"
            )

    report.append("")
    report.append("─" * 70)
    report.append(f"📈 Pewność identyfikacji: ", )

    score = best["final_score"]
    if score > 0.9:
        report[-1] += "WYSOKA ✅ — Model zidentyfikowany jednoznacznie"
    elif score > 0.7:
        report[-1] += "ŚREDNIA ⚠️ — Prawdopodobne dopasowanie, zalecana weryfikacja"
    elif score > 0.5:
        report[-1] += "NISKA ⚠️ — Wymagana weryfikacja manualna"
    else:
        report[-1] += "BARDZO NISKA ❌ — Model nierozpoznany"

    if len(results) >= 2:
        gap = results[0]["final_score"] - results[1]["final_score"]
        if gap < 0.05:
            report.append("⚠️  Uwaga: Dwa najlepsze wyniki są bardzo bliskie! "
                          "Zalecana ręczna weryfikacja.")

    report.append("=" * 70)

    return "\n".join(report)


# ============================================================================
# 9. API — UPROSZCZONY INTERFEJS
# ============================================================================


def identify_from_scan(
    scan_source,
    db_path: str = "models_3d.db",
    top_k: int = 5,
    category_filter: Optional[str] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Jedna funkcja do identyfikacji modelu na podstawie skanu.

    Args:
        scan_source: Ścieżka do pliku ze skanem, lub numpy array (N, 3)
        db_path: Ścieżka do bazy modeli
        top_k: Ile najlepszych wyników
        category_filter: Opcjonalny filtr kategorii
        verbose: Czy wyświetlać informacje

    Returns:
        Lista wyników z identyfikacji

    Przykład:
        results = identify_from_scan("scan_output.ply")
        results = identify_from_scan(numpy_points, category_filter="fasteners")
    """
    db = Model3DDatabase(db_path)
    comparator = Model3DComparator(db, verbose=verbose)

    if isinstance(scan_source, np.ndarray):
        scan_points = scan_source
    else:
        processor = ScannerDataProcessor(verbose=verbose)
        processor.load(scan_source)
        scan_points = processor.points

    results = comparator.identify(
        scan_points,
        top_k=top_k,
        category_filter=category_filter,
    )

    return results


# ============================================================================
# INFO / WERSJA
# ============================================================================

__version__ = "1.0.0"
__all__ = [
    "Model3DDescriptor",
    "Model3DDatabase",
    "Model3DRecord",
    "ComparisonLog",
    "ScannerDataProcessor",
    "Model3DComparator",
    "extract_descriptor_from_file",
    "generate_demo_3d_models",
    "generate_demo_scan",
    "identify_from_scan",
    "plot_identification_results",
    "plot_point_clouds_comparison",
    "generate_identification_report",
]

if __name__ == "__main__":
    print(f"Scanner 3D Comparator v{__version__}")
    print("Uruchom: generate_demo_3d_models() aby wygenerować dane testowe")
    print("Uruchom: identify_from_scan(scan_data) aby zidentyfikować model")
