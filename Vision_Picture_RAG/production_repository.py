"""
Punkt 2: Repozytorium danych produkcyjnych z systemem wersjonowania

Moduł zapewnia:
- Zarządzanie modelami 3D (CAD) z wersjami
- Programy obróbcze (G-code, post-processor config)
- Dane techniczne (rysunki, specyfikacje, parametry)
- System wersjonowania z pełną historią zmian
- Archiwizację konfiguracji produkcyjnych
- Tracking zależności między artefaktami

Architektura:
    ProductionRepository
    ├── CADModelVersion (modele 3D z historią)
    ├── MachiningProgramVersion (G-code z parametrami)
    ├── TechnicalDocumentVersion (rysunki, spec)
    ├── ProductionConfigVersion (parametry obróbki)
    └── ChangeLog (audit trail)
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    ForeignKey, Text, Boolean, JSON, LargeBinary, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

# ============================================================================
# 1. TYPY I ENUMY
# ============================================================================

BaseRepo = declarative_base()


class ArtifactType(str, Enum):
    """Typy artefaktów w repozytorium"""
    CAD_MODEL = "cad_model"                # Modele 3D (STEP, STL, IGES)
    MACHINING_PROGRAM = "machining_program"  # Programy CNC (G-code, ISO)
    TECHNICAL_DRAWING = "technical_drawing"  # Rysunki techniczne (PDF, DWG)
    SPECIFICATION = "specification"         # Specyfikacje (PDF, TXT, MD)
    PRODUCTION_CONFIG = "production_config"  # Parametry produkcyjne (JSON)
    MATERIAL_SPEC = "material_spec"         # Specyfikacja materiału
    TOOL_LIST = "tool_list"                 # Lista narzędzi


class ChangeType(str, Enum):
    """Typy zmian w repozytorium"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"
    FORK = "fork"


class ApprovalStatus(str, Enum):
    """Status zatwierdzenia wersji"""
    DRAFT = "draft"              # Wersja robocza
    REVIEW = "review"            # W trakcie przeglądu
    APPROVED = "approved"        # Zatwierdzona
    PRODUCTION = "production"    # W produkcji
    DEPRECATED = "deprecated"    # Przestarzała
    ARCHIVED = "archived"        # Zarchiwizowana


# ============================================================================
# 2. ORM MODELS — WERSJONOWANIE
# ============================================================================

class CADModelVersion(BaseRepo):
    """
    Wersjonowanie modeli CAD (STEP, STL, IGES, Parasolid).
    Każda zmiana geometrii = nowa wersja.
    """
    __tablename__ = "cad_model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identyfikacja
    model_id = Column(String(100), nullable=False, index=True)  # Unikalny ID modelu (np. "BOLT-M8x20")
    version = Column(Integer, nullable=False)  # Numer wersji (1, 2, 3...)
    version_tag = Column(String(50))           # Tag wersji (np. "v1.2.0", "PROD-2024-01")
    
    # Metadata podstawowe
    model_name = Column(String(200), nullable=False)
    description = Column(Text)
    category = Column(String(100))  # fasteners, bearings, shafts, plates...
    part_number = Column(String(100), index=True)  # Numer katalogowy
    
    # Plik CAD
    file_path = Column(String(500))           # Relatywna ścieżka do pliku
    file_format = Column(String(20))          # STEP, STL, IGES, SAT
    file_size_bytes = Column(Integer)
    file_hash_sha256 = Column(String(64), index=True)  # Hash dla integrity checking
    
    # Geometria — metadane
    bounding_box_json = Column(JSON)  # {"min": [x,y,z], "max": [x,y,z]}
    volume_mm3 = Column(Float)
    surface_area_mm2 = Column(Float)
    mass_grams = Column(Float)  # Jeśli znana gęstość materiału
    
    # Wersjonowanie
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100))  # User/system ID
    parent_version_id = Column(Integer, ForeignKey("cad_model_versions.id"), nullable=True)
    is_latest = Column(Boolean, default=True)  # Czy to najnowsza wersja modelu
    
    # Status
    approval_status = Column(String(20), default=ApprovalStatus.DRAFT.value)
    approved_by = Column(String(100))
    approved_at = Column(DateTime)
    
    # Notatki i changelog
    change_notes = Column(Text)  # Co się zmieniło w tej wersji
    tags_json = Column(JSON)     # Tagi: ["critical", "customer-xyz", "revision-A"]
    
    # Relacje
    parent = relationship("CADModelVersion", remote_side=[id], backref="children")
    machining_programs = relationship("MachiningProgramVersion", back_populates="cad_model")
    technical_docs = relationship("TechnicalDocumentVersion", back_populates="cad_model")
    
    __table_args__ = (
        Index("idx_model_version", "model_id", "version"),
        Index("idx_status", "approval_status"),
    )


class MachiningProgramVersion(BaseRepo):
    """
    Wersjonowanie programów obróbczych (G-code, ISO, Heidenhain, Mazak, Fanuc).
    Połączone z konkretną wersją modelu CAD.
    """
    __tablename__ = "machining_program_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identyfikacja
    program_id = Column(String(100), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    version_tag = Column(String(50))
    
    # Powiązanie z CAD
    cad_model_version_id = Column(Integer, ForeignKey("cad_model_versions.id"), nullable=False)
    
    # Metadata
    program_name = Column(String(200), nullable=False)
    description = Column(Text)
    machine_type = Column(String(100))  # "3-axis mill", "5-axis mill", "lathe"
    controller_type = Column(String(100))  # "Fanuc 31i", "Siemens 840D", "Heidenhain TNC640"
    post_processor = Column(String(100))  # Nazwa post-processora
    
    # Plik programu
    file_path = Column(String(500))
    file_format = Column(String(20))  # "gcode", "nc", "tap", "mpf"
    file_size_bytes = Column(Integer)
    file_hash_sha256 = Column(String(64), index=True)
    
    # Parametry obróbki
    setup_time_minutes = Column(Float)
    cycle_time_minutes = Column(Float)
    spindle_speed_rpm = Column(Integer)
    feed_rate_mmpm = Column(Float)
    tool_count = Column(Integer)
    
    # Narzędzia (referencja do listy narzędzi)
    tool_list_json = Column(JSON)  # [{"tool_no": 1, "description": "End mill 10mm", "diameter": 10}, ...]
    
    # Parametry produkcyjne
    production_config_json = Column(JSON)  # Coolant, roughness, tolerances...
    
    # Wersjonowanie
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100))
    parent_version_id = Column(Integer, ForeignKey("machining_program_versions.id"), nullable=True)
    is_latest = Column(Boolean, default=True)
    
    # Status
    approval_status = Column(String(20), default=ApprovalStatus.DRAFT.value)
    approved_by = Column(String(100))
    approved_at = Column(DateTime)
    simulation_verified = Column(Boolean, default=False)  # Czy zweryfikowano w symulatorze
    
    # Changelog
    change_notes = Column(Text)
    tags_json = Column(JSON)
    
    # Relacje
    cad_model = relationship("CADModelVersion", back_populates="machining_programs")
    parent = relationship("MachiningProgramVersion", remote_side=[id], backref="children")
    
    __table_args__ = (
        Index("idx_program_version", "program_id", "version"),
    )


class TechnicalDocumentVersion(BaseRepo):
    """
    Wersjonowanie dokumentacji technicznej (rysunki 2D, specyfikacje, karty technologiczne).
    """
    __tablename__ = "technical_document_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identyfikacja
    document_id = Column(String(100), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    version_tag = Column(String(50))
    
    # Powiązanie z CAD
    cad_model_version_id = Column(Integer, ForeignKey("cad_model_versions.id"), nullable=True)
    
    # Metadata
    document_name = Column(String(200), nullable=False)
    document_type = Column(String(50))  # "technical_drawing", "specification", "process_sheet"
    description = Column(Text)
    
    # Plik
    file_path = Column(String(500))
    file_format = Column(String(20))  # PDF, DWG, DXF, MD, TXT
    file_size_bytes = Column(Integer)
    file_hash_sha256 = Column(String(64), index=True)
    
    # Zawartość
    standard = Column(String(100))  # ISO 128, ASME Y14.5, DIN...
    revision_mark = Column(String(20))  # "A", "B", "C"...
    sheet_count = Column(Integer)
    
    # Dane techniczne (JSON dla elastyczności)
    technical_data_json = Column(JSON)  # Tolerances, surface finish, hardness, etc.
    
    # Wersjonowanie
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100))
    parent_version_id = Column(Integer, ForeignKey("technical_document_versions.id"), nullable=True)
    is_latest = Column(Boolean, default=True)
    
    # Status
    approval_status = Column(String(20), default=ApprovalStatus.DRAFT.value)
    approved_by = Column(String(100))
    approved_at = Column(DateTime)
    
    # Changelog
    change_notes = Column(Text)
    tags_json = Column(JSON)
    
    # Relacje
    cad_model = relationship("CADModelVersion", back_populates="technical_docs")
    parent = relationship("TechnicalDocumentVersion", remote_side=[id], backref="children")
    
    __table_args__ = (
        Index("idx_document_version", "document_id", "version"),
    )


class ProductionConfigVersion(BaseRepo):
    """
    Wersjonowanie konfiguracji produkcyjnych (parametry maszyn, setupy, fixtures).
    Archiwizacja zmian parametrów w czasie.
    """
    __tablename__ = "production_config_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Identyfikacja
    config_id = Column(String(100), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    version_tag = Column(String(50))
    
    # Metadata
    config_name = Column(String(200), nullable=False)
    description = Column(Text)
    config_type = Column(String(50))  # "machine_setup", "fixture", "tooling", "quality_params"
    
    # Powiązanie (opcjonalne)
    cad_model_version_id = Column(Integer, ForeignKey("cad_model_versions.id"), nullable=True)
    machining_program_id = Column(Integer, ForeignKey("machining_program_versions.id"), nullable=True)
    
    # Konfiguracja jako JSON
    config_data_json = Column(JSON, nullable=False)
    """
    Przykład:
    {
        "machine": "DMC 80U duoBLOCK",
        "spindle_max_rpm": 18000,
        "coolant_type": "emulsion 5%",
        "work_offset": "G54",
        "fixture_id": "FIX-001",
        "quality_checks": {
            "roughness_Ra": {"min": 0.0, "max": 1.6, "unit": "μm"},
            "tolerance_IT": "IT7"
        }
    }
    """
    
    # Hash dla szybkiego porównania
    config_hash_sha256 = Column(String(64), index=True)
    
    # Wersjonowanie
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100))
    parent_version_id = Column(Integer, ForeignKey("production_config_versions.id"), nullable=True)
    is_latest = Column(Boolean, default=True)
    
    # Status
    approval_status = Column(String(20), default=ApprovalStatus.DRAFT.value)
    approved_by = Column(String(100))
    approved_at = Column(DateTime)
    active_in_production = Column(Boolean, default=False)
    
    # Changelog
    change_notes = Column(Text)
    tags_json = Column(JSON)
    
    # Relacje
    parent = relationship("ProductionConfigVersion", remote_side=[id], backref="children")
    
    __table_args__ = (
        Index("idx_config_version", "config_id", "version"),
    )


class ChangeLog(BaseRepo):
    """
    Audit trail — każda zmiana w repozytorium zapisywana w logu.
    """
    __tablename__ = "change_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    
    # Typ operacji
    change_type = Column(String(20), nullable=False)  # CREATE, UPDATE, DELETE, RESTORE, FORK
    artifact_type = Column(String(50), nullable=False)  # CAD_MODEL, MACHINING_PROGRAM, etc.
    
    # Odniesienie do artefaktu
    artifact_id = Column(String(100), nullable=False, index=True)  # model_id / program_id / ...
    version = Column(Integer)
    record_id = Column(Integer)  # Foreign key do tabeli wersji
    
    # Szczegóły zmiany
    description = Column(Text)
    diff_json = Column(JSON)  # Zmiany: {"field": {"old": ..., "new": ...}}
    metadata_json = Column(JSON)  # IP, hostname, etc.
    
    __table_args__ = (
        Index("idx_artifact_changes", "artifact_id", "timestamp"),
        Index("idx_user_changes", "user_id", "timestamp"),
    )


# ============================================================================
# 3. KLASA REPOZYTORIUM — GŁÓWNA LOGIKA
# ============================================================================

class ProductionRepository:
    """
    Główny interfejs do zarządzania repozytorium produkcyjnym.
    
    Funkcjonalności:
    - Dodawanie nowych artefaktów (CAD, G-code, dokumenty)
    - Tworzenie nowych wersji (update)
    - Zarządzanie statusami (draft → review → approved → production)
    - Przeglądanie historii zmian
    - Przywracanie starych wersji (restore)
    - Archiwizacja i cleanup
    """
    
    def __init__(self, db_path: str = "production_repo.db", storage_root: str = "./repo_storage"):
        """
        Args:
            db_path: Ścieżka do bazy SQLite
            storage_root: Katalog główny dla plików (CAD, G-code, PDF, etc.)
        """
        self.db_path = db_path
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        
        # SQLAlchemy setup
        self.engine = create_engine(f"sqlite:///{db_path}")
        BaseRepo.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    # ---- Utility methods ----
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Oblicz SHA-256 hash pliku."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _compute_json_hash(self, data: dict) -> str:
        """Oblicz hash JSON (dla config versioning)."""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _store_file(self, source_path: str, artifact_type: str, artifact_id: str, version: int) -> Tuple[str, int]:
        """
        Kopiuj plik do struktury repozytorium.
        
        Returns:
            (relative_path, file_size_bytes)
        """
        ext = Path(source_path).suffix
        rel_dir = Path(artifact_type) / artifact_id / f"v{version}"
        dest_dir = self.storage_root / rel_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        filename = Path(source_path).name
        dest_path = dest_dir / filename
        shutil.copy2(source_path, dest_path)
        
        file_size = dest_path.stat().st_size
        relative_path = str(rel_dir / filename)
        
        return relative_path, file_size
    
    def _log_change(self, session: Session, change_type: ChangeType, artifact_type: ArtifactType,
                    artifact_id: str, version: int, record_id: int, user_id: str, description: str,
                    diff: Optional[dict] = None):
        """Dodaj wpis do change logu."""
        log_entry = ChangeLog(
            user_id=user_id,
            change_type=change_type.value,
            artifact_type=artifact_type.value,
            artifact_id=artifact_id,
            version=version,
            record_id=record_id,
            description=description,
            diff_json=diff,
        )
        session.add(log_entry)
    
    # ---- CAD Model management ----
    
    def add_cad_model(
        self,
        model_id: str,
        model_name: str,
        file_path: str,
        category: str = "",
        part_number: str = "",
        description: str = "",
        created_by: str = "system",
        **kwargs
    ) -> CADModelVersion:
        """
        Dodaj nowy model CAD (wersja 1).
        
        Args:
            model_id: Unikalny ID (np. "BOLT-M8x20")
            model_name: Nazwa opisowa
            file_path: Ścieżka do pliku CAD (STEP/STL/IGES)
            category: Kategoria (fasteners, bearings, ...)
            part_number: Numer katalogowy
            description: Opis
            created_by: User ID
            **kwargs: Dodatkowe parametry (volume_mm3, mass_grams, tags_json, ...)
        
        Returns:
            CADModelVersion record
        """
        with self.SessionLocal() as session:
            # Sprawdź czy model już istnieje
            existing = session.query(CADModelVersion).filter_by(model_id=model_id).first()
            if existing:
                raise ValueError(f"Model {model_id} już istnieje. Użyj update_cad_model().")
            
            # Hash pliku
            file_hash = self._compute_file_hash(file_path)
            
            # Przechowaj plik
            file_format = Path(file_path).suffix.lstrip(".").upper()
            rel_path, file_size = self._store_file(file_path, ArtifactType.CAD_MODEL.value, model_id, version=1)
            
            # Utwórz rekord
            cad_model = CADModelVersion(
                model_id=model_id,
                version=1,
                model_name=model_name,
                description=description,
                category=category,
                part_number=part_number,
                file_path=rel_path,
                file_format=file_format,
                file_size_bytes=file_size,
                file_hash_sha256=file_hash,
                created_by=created_by,
                is_latest=True,
                **kwargs
            )
            session.add(cad_model)
            session.flush()
            
            # Log
            self._log_change(
                session, ChangeType.CREATE, ArtifactType.CAD_MODEL,
                model_id, 1, cad_model.id, created_by,
                f"Utworzono model CAD: {model_name}"
            )
            
            session.commit()
            session.refresh(cad_model)
            return cad_model
    
    def update_cad_model(
        self,
        model_id: str,
        file_path: str,
        change_notes: str,
        created_by: str = "system",
        version_tag: Optional[str] = None,
        **kwargs
    ) -> CADModelVersion:
        """
        Utwórz nową wersję modelu CAD.
        
        Args:
            model_id: ID modelu do aktualizacji
            file_path: Nowa wersja pliku CAD
            change_notes: Opis zmian
            created_by: User ID
            version_tag: Opcjonalny tag (np. "v2.0.0")
            **kwargs: Nadpisanie metadanych (model_name, description, ...)
        
        Returns:
            Nowa wersja CADModelVersion
        """
        with self.SessionLocal() as session:
            # Znajdź ostatnią wersję
            latest = session.query(CADModelVersion).filter_by(
                model_id=model_id, is_latest=True
            ).first()
            
            if not latest:
                raise ValueError(f"Model {model_id} nie istnieje. Użyj add_cad_model().")
            
            # Hash pliku
            file_hash = self._compute_file_hash(file_path)
            
            # Sprawdź czy coś się zmieniło
            if file_hash == latest.file_hash_sha256:
                print(f"⚠️ Plik identyczny jak wersja {latest.version}. Brak zmian geometrycznych.")
                return latest
            
            # Nowa wersja
            new_version_num = latest.version + 1
            file_format = Path(file_path).suffix.lstrip(".").upper()
            rel_path, file_size = self._store_file(file_path, ArtifactType.CAD_MODEL.value, model_id, new_version_num)
            
            # Oznacz previous jako not latest
            latest.is_latest = False
            
            # Utwórz nowy rekord (inherit metadata)
            new_version = CADModelVersion(
                model_id=model_id,
                version=new_version_num,
                version_tag=version_tag,
                model_name=kwargs.get("model_name", latest.model_name),
                description=kwargs.get("description", latest.description),
                category=kwargs.get("category", latest.category),
                part_number=kwargs.get("part_number", latest.part_number),
                file_path=rel_path,
                file_format=file_format,
                file_size_bytes=file_size,
                file_hash_sha256=file_hash,
                created_by=created_by,
                parent_version_id=latest.id,
                is_latest=True,
                change_notes=change_notes,
                approval_status=kwargs.get("approval_status", ApprovalStatus.DRAFT.value),
                tags_json=kwargs.get("tags_json", latest.tags_json),
            )
            
            # Dodatkowe parametry
            for key, value in kwargs.items():
                if hasattr(new_version, key):
                    setattr(new_version, key, value)
            
            session.add(new_version)
            session.flush()
            
            # Log
            self._log_change(
                session, ChangeType.UPDATE, ArtifactType.CAD_MODEL,
                model_id, new_version_num, new_version.id, created_by,
                f"Aktualizacja modelu CAD → v{new_version_num}: {change_notes}"
            )
            
            session.commit()
            session.refresh(new_version)
            return new_version
    
    def get_cad_model_latest(self, model_id: str) -> Optional[CADModelVersion]:
        """Pobierz najnowszą wersję modelu CAD."""
        with self.SessionLocal() as session:
            return session.query(CADModelVersion).filter_by(
                model_id=model_id, is_latest=True
            ).first()
    
    def get_cad_model_version(self, model_id: str, version: int) -> Optional[CADModelVersion]:
        """Pobierz konkretną wersję modelu CAD."""
        with self.SessionLocal() as session:
            return session.query(CADModelVersion).filter_by(
                model_id=model_id, version=version
            ).first()
    
    def get_cad_model_history(self, model_id: str) -> List[CADModelVersion]:
        """Pobierz pełną historię wersji modelu CAD."""
        with self.SessionLocal() as session:
            return session.query(CADModelVersion).filter_by(
                model_id=model_id
            ).order_by(CADModelVersion.version).all()
    
    # ---- Machining Program management ----
    
    def add_machining_program(
        self,
        program_id: str,
        program_name: str,
        file_path: str,
        cad_model_id: str,
        cad_version: Optional[int] = None,
        machine_type: str = "",
        controller_type: str = "",
        created_by: str = "system",
        **kwargs
    ) -> MachiningProgramVersion:
        """Dodaj nowy program obróbczy (wersja 1)."""
        with self.SessionLocal() as session:
            # Znajdź CAD model
            if cad_version:
                cad_model = self.get_cad_model_version(cad_model_id, cad_version)
            else:
                cad_model = self.get_cad_model_latest(cad_model_id)
            
            if not cad_model:
                raise ValueError(f"CAD model {cad_model_id} (v{cad_version}) nie istnieje.")
            
            # Hash pliku
            file_hash = self._compute_file_hash(file_path)
            file_format = Path(file_path).suffix.lstrip(".")
            rel_path, file_size = self._store_file(file_path, ArtifactType.MACHINING_PROGRAM.value, program_id, 1)
            
            program = MachiningProgramVersion(
                program_id=program_id,
                version=1,
                program_name=program_name,
                cad_model_version_id=cad_model.id,
                file_path=rel_path,
                file_format=file_format,
                file_size_bytes=file_size,
                file_hash_sha256=file_hash,
                machine_type=machine_type,
                controller_type=controller_type,
                created_by=created_by,
                is_latest=True,
                **kwargs
            )
            session.add(program)
            session.flush()
            
            self._log_change(
                session, ChangeType.CREATE, ArtifactType.MACHINING_PROGRAM,
                program_id, 1, program.id, created_by,
                f"Utworzono program CNC: {program_name}"
            )
            
            session.commit()
            session.refresh(program)
            return program
    
    def update_machining_program(
        self,
        program_id: str,
        file_path: str,
        change_notes: str,
        created_by: str = "system",
        **kwargs
    ) -> MachiningProgramVersion:
        """Utwórz nową wersję programu obróbczego."""
        with self.SessionLocal() as session:
            latest = session.query(MachiningProgramVersion).filter_by(
                program_id=program_id, is_latest=True
            ).first()
            
            if not latest:
                raise ValueError(f"Program {program_id} nie istnieje.")
            
            file_hash = self._compute_file_hash(file_path)
            if file_hash == latest.file_hash_sha256:
                print(f"⚠️ Plik identyczny jak wersja {latest.version}.")
                return latest
            
            new_version_num = latest.version + 1
            file_format = Path(file_path).suffix.lstrip(".")
            rel_path, file_size = self._store_file(file_path, ArtifactType.MACHINING_PROGRAM.value, program_id, new_version_num)
            
            latest.is_latest = False
            
            new_version = MachiningProgramVersion(
                program_id=program_id,
                version=new_version_num,
                program_name=kwargs.get("program_name", latest.program_name),
                cad_model_version_id=kwargs.get("cad_model_version_id", latest.cad_model_version_id),
                file_path=rel_path,
                file_format=file_format,
                file_size_bytes=file_size,
                file_hash_sha256=file_hash,
                machine_type=kwargs.get("machine_type", latest.machine_type),
                controller_type=kwargs.get("controller_type", latest.controller_type),
                created_by=created_by,
                parent_version_id=latest.id,
                is_latest=True,
                change_notes=change_notes,
                **{k: v for k, v in kwargs.items() if hasattr(MachiningProgramVersion, k)}
            )
            
            session.add(new_version)
            session.flush()
            
            self._log_change(
                session, ChangeType.UPDATE, ArtifactType.MACHINING_PROGRAM,
                program_id, new_version_num, new_version.id, created_by,
                f"Aktualizacja programu CNC → v{new_version_num}: {change_notes}"
            )
            
            session.commit()
            session.refresh(new_version)
            return new_version
    
    # ---- Production Config versioning ----
    
    def add_production_config(
        self,
        config_id: str,
        config_name: str,
        config_data: dict,
        config_type: str = "machine_setup",
        created_by: str = "system",
        **kwargs
    ) -> ProductionConfigVersion:
        """Dodaj konfigurację produkcyjną (wersja 1)."""
        with self.SessionLocal() as session:
            config_hash = self._compute_json_hash(config_data)
            
            config = ProductionConfigVersion(
                config_id=config_id,
                version=1,
                config_name=config_name,
                config_type=config_type,
                config_data_json=config_data,
                config_hash_sha256=config_hash,
                created_by=created_by,
                is_latest=True,
                **kwargs
            )
            session.add(config)
            session.flush()
            
            self._log_change(
                session, ChangeType.CREATE, ArtifactType.PRODUCTION_CONFIG,
                config_id, 1, config.id, created_by,
                f"Utworzono konfigurację: {config_name}"
            )
            
            session.commit()
            session.refresh(config)
            return config
    
    def update_production_config(
        self,
        config_id: str,
        config_data: dict,
        change_notes: str,
        created_by: str = "system",
        **kwargs
    ) -> ProductionConfigVersion:
        """Utwórz nową wersję konfiguracji produkcyjnej."""
        with self.SessionLocal() as session:
            latest = session.query(ProductionConfigVersion).filter_by(
                config_id=config_id, is_latest=True
            ).first()
            
            if not latest:
                raise ValueError(f"Config {config_id} nie istnieje.")
            
            config_hash = self._compute_json_hash(config_data)
            if config_hash == latest.config_hash_sha256:
                print(f"⚠️ Konfiguracja identyczna jak wersja {latest.version}.")
                return latest
            
            # Oblicz diff
            diff = self._compute_config_diff(latest.config_data_json, config_data)
            
            new_version_num = latest.version + 1
            latest.is_latest = False
            
            new_version = ProductionConfigVersion(
                config_id=config_id,
                version=new_version_num,
                config_name=kwargs.get("config_name", latest.config_name),
                config_type=kwargs.get("config_type", latest.config_type),
                config_data_json=config_data,
                config_hash_sha256=config_hash,
                created_by=created_by,
                parent_version_id=latest.id,
                is_latest=True,
                change_notes=change_notes,
                **{k: v for k, v in kwargs.items() if hasattr(ProductionConfigVersion, k)}
            )
            
            session.add(new_version)
            session.flush()
            
            self._log_change(
                session, ChangeType.UPDATE, ArtifactType.PRODUCTION_CONFIG,
                config_id, new_version_num, new_version.id, created_by,
                f"Aktualizacja config → v{new_version_num}: {change_notes}",
                diff=diff
            )
            
            session.commit()
            session.refresh(new_version)
            return new_version
    
    def _compute_config_diff(self, old: dict, new: dict) -> dict:
        """Oblicz różnice między dwoma konfiguracjami JSON."""
        diff = {}
        all_keys = set(old.keys()) | set(new.keys())
        
        for key in all_keys:
            old_val = old.get(key)
            new_val = new.get(key)
            if old_val != new_val:
                diff[key] = {"old": old_val, "new": new_val}
        
        return diff
    
    # ---- Change log & audit ----
    
    def get_change_log(
        self,
        artifact_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ChangeLog]:
        """Pobierz historię zmian."""
        with self.SessionLocal() as session:
            q = session.query(ChangeLog).order_by(ChangeLog.timestamp.desc())
            
            if artifact_id:
                q = q.filter_by(artifact_id=artifact_id)
            if user_id:
                q = q.filter_by(user_id=user_id)
            
            return q.limit(limit).all()
    
    def get_repository_stats(self) -> dict:
        """Statystyki repozytorium."""
        with self.SessionLocal() as session:
            cad_count = session.query(CADModelVersion).filter_by(is_latest=True).count()
            prog_count = session.query(MachiningProgramVersion).filter_by(is_latest=True).count()
            config_count = session.query(ProductionConfigVersion).filter_by(is_latest=True).count()
            total_versions = (
                session.query(CADModelVersion).count()
                + session.query(MachiningProgramVersion).count()
                + session.query(ProductionConfigVersion).count()
            )
            
            return {
                "cad_models_latest": cad_count,
                "machining_programs_latest": prog_count,
                "production_configs_latest": config_count,
                "total_versions_all": total_versions,
            }
    
    # ---- Approval workflow ----
    
    def approve_version(
        self,
        artifact_type: ArtifactType,
        artifact_id: str,
        version: int,
        approved_by: str,
        target_status: ApprovalStatus = ApprovalStatus.APPROVED
    ):
        """Zatwierdź wersję (zmień status)."""
        with self.SessionLocal() as session:
            if artifact_type == ArtifactType.CAD_MODEL:
                record = session.query(CADModelVersion).filter_by(
                    model_id=artifact_id, version=version
                ).first()
            elif artifact_type == ArtifactType.MACHINING_PROGRAM:
                record = session.query(MachiningProgramVersion).filter_by(
                    program_id=artifact_id, version=version
                ).first()
            elif artifact_type == ArtifactType.PRODUCTION_CONFIG:
                record = session.query(ProductionConfigVersion).filter_by(
                    config_id=artifact_id, version=version
                ).first()
            else:
                raise ValueError(f"Unsupported artifact type: {artifact_type}")
            
            if not record:
                raise ValueError(f"{artifact_type} {artifact_id} v{version} nie istnieje.")
            
            old_status = record.approval_status
            record.approval_status = target_status.value
            record.approved_by = approved_by
            record.approved_at = datetime.utcnow()
            
            self._log_change(
                session, ChangeType.UPDATE, artifact_type,
                artifact_id, version, record.id, approved_by,
                f"Zmiana statusu: {old_status} → {target_status.value}"
            )
            
            session.commit()
            print(f"✅ {artifact_type.value} {artifact_id} v{version} → {target_status.value}")


# ============================================================================
# 4. HELPER FUNCTIONS
# ============================================================================

def generate_demo_repository(repo_path: str = "./demo_production_repo") -> ProductionRepository:
    """
    Generuje demonstracyjne repozytorium z przykładowymi danymi.
    """
    import tempfile
    
    repo = ProductionRepository(
        db_path=f"{repo_path}/production_repo.db",
        storage_root=f"{repo_path}/storage"
    )
    
    print("📦 Generowanie demonstracyjnego repozytorium produkcyjnego...")
    
    # ---- Przykładowy model CAD ----
    with tempfile.NamedTemporaryFile(mode="w", suffix=".step", delete=False) as f:
        f.write("ISO-10303-21; HEADER; ... END-ISO-10303-21;")  # Dummy STEP
        cad_file_v1 = f.name
    
    cad_v1 = repo.add_cad_model(
        model_id="DEMO-BOLT-M10",
        model_name="Śruba demonstracyjna M10x50",
        file_path=cad_file_v1,
        category="fasteners",
        part_number="BOLT-M10-50-A2",
        description="Śruba sześciokątna M10x50 DIN 933 A2-70",
        volume_mm3=850.5,
        mass_grams=6.7,
        created_by="jan.kowalski",
        tags_json=["demo", "bolt", "DIN933"]
    )
    print(f"   ✅ CAD model v1: {cad_v1.model_name}")
    
    # ---- Aktualizacja modelu (v2) ----
    with tempfile.NamedTemporaryFile(mode="w", suffix=".step", delete=False) as f:
        f.write("ISO-10303-21; HEADER; ... UPDATED GEOMETRY ... END-ISO-10303-21;")
        cad_file_v2 = f.name
    
    cad_v2 = repo.update_cad_model(
        model_id="DEMO-BOLT-M10",
        file_path=cad_file_v2,
        change_notes="Poprawiono średnicę trzpienia z 9.8mm na 10.0mm zgodnie z DIN 933",
        volume_mm3=860.2,
        created_by="anna.nowak",
        version_tag="v2.0.0"
    )
    print(f"   ✅ CAD model v2: {cad_v2.version_tag}")
    
    # ---- Program obróbczy ----
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nc", delete=False) as f:
        f.write("G0 G90 G40 G49 G80\nG54\nM6 T1\nS1200 M3\nG43 H1 Z100\n...")
        gcode_v1 = f.name
    
    prog_v1 = repo.add_machining_program(
        program_id="DEMO-BOLT-M10-MILL",
        program_name="Frezowanie śruby M10 - 3-osiowe",
        file_path=gcode_v1,
        cad_model_id="DEMO-BOLT-M10",
        machine_type="3-axis mill",
        controller_type="Fanuc 31i",
        post_processor="Fusion360_Fanuc",
        cycle_time_minutes=4.5,
        tool_count=3,
        tool_list_json=[
            {"tool_no": 1, "description": "Frez walcowy 12mm", "diameter": 12},
            {"tool_no": 2, "description": "Frez trzpieniowy 6mm", "diameter": 6},
            {"tool_no": 3, "description": "Gwintownik M10", "diameter": 10}
        ],
        created_by="jan.kowalski"
    )
    print(f"   ✅ Program CNC v1: {prog_v1.program_name}")
    
    # ---- Konfiguracja produkcyjna ----
    config_v1 = repo.add_production_config(
        config_id="DEMO-MILL-SETUP-001",
        config_name="Setup frezowania śrub M10 na DMC80",
        config_type="machine_setup",
        config_data={
            "machine": "DMC 80U duoBLOCK",
            "spindle_max_rpm": 18000,
            "coolant_type": "emulsion 5%",
            "work_offset": "G54",
            "fixture_id": "VISE-125mm",
            "clamping_force_N": 5000,
            "quality_params": {
                "tolerance_IT": "IT7",
                "roughness_Ra_max_um": 1.6,
                "inspection_frequency": "every 10 parts"
            }
        },
        created_by="jan.kowalski"
    )
    print(f"   ✅ Config v1: {config_v1.config_name}")
    
    # ---- Update config ----
    config_v2 = repo.update_production_config(
        config_id="DEMO-MILL-SETUP-001",
        config_data={
            "machine": "DMC 80U duoBLOCK",
            "spindle_max_rpm": 18000,
            "coolant_type": "emulsion 5%",
            "work_offset": "G54",
            "fixture_id": "VISE-125mm",
            "clamping_force_N": 6000,  # ZMIANA
            "quality_params": {
                "tolerance_IT": "IT7",
                "roughness_Ra_max_um": 1.2,  # ZMIANA
                "inspection_frequency": "every 5 parts"  # ZMIANA
            }
        },
        change_notes="Zwiększono siłę docisku i zaostrzone wymagania jakościowe",
        created_by="anna.nowak"
    )
    print(f"   ✅ Config v2: aktualizacja parametrów")
    
    # ---- Approval workflow ----
    repo.approve_version(
        ArtifactType.CAD_MODEL, "DEMO-BOLT-M10", 2,
        approved_by="kierownik.produkcji",
        target_status=ApprovalStatus.APPROVED
    )
    
    repo.approve_version(
        ArtifactType.MACHINING_PROGRAM, "DEMO-BOLT-M10-MILL", 1,
        approved_by="kierownik.produkcji",
        target_status=ApprovalStatus.PRODUCTION
    )
    
    print()
    print("📊 Statystyki repozytorium:")
    stats = repo.get_repository_stats()
    for key, val in stats.items():
        print(f"   {key}: {val}")
    
    return repo


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    repo = generate_demo_repository()
    
    print("\n" + "=" * 80)
    print("📜 CHANGE LOG (ostatnie 10 zmian)")
    print("=" * 80)
    
    logs = repo.get_change_log(limit=10)
    for log in logs:
        print(f"[{log.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"{log.user_id} | {log.change_type} | "
              f"{log.artifact_type} {log.artifact_id} v{log.version}")
        print(f"   → {log.description}")
        if log.diff_json:
            print(f"   Diff: {log.diff_json}")
    
    print("\n✅ Repozytorium produkcyjne z wersjonowaniem gotowe!")
