"""
Punkt 6: Algorytmy obsługujące proces produkcyjny

Zakres:
- Etapy procesu: INPUT → SKANOWANIE → IDENTYFIKACJA → ROUTING → OBRÓBKA → QC → ARCHIWUM
- Warianty i ścieżki warunkowe (zależne od typu części, błędów, wyników QC)
- Integracja z: process_flow_decision, scanner_3d_comparator, production_repository, qwen_image_verifier
- Śledzenie statusu części i logów przejść

Architektura:
    ProcessWorkflow
    ├── Stage (enum): INPUT, SCAN, IDENTIFY, ROUTE, MACHINE, QC, ARCHIVE
    ├── PartJob (dataclass): ID części, stage, metadata, logging
    ├── WorkflowEngine: executors dla każdego etapu
    └── WorkflowLog: audit trail przejść między etapami
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Callable, Tuple, Any
from datetime import datetime
from pathlib import Path
import json
import logging

# ============================================================================
# 1. ENUMERACJE I TYPY
# ============================================================================

class ProcessStage(str, Enum):
    """Etapy procesu produkcyjnego."""
    INPUT = "input"                    # Część przychodzi do systemu
    SCAN_3D = "scan_3d"               # Skanowanie 3D
    IDENTIFY = "identify"              # Identyfikacja modelu
    ROUTE = "route"                    # Routing: jaką ścieżką obróbki?
    MACHINE = "machine"                # Obróbka CNC / inne
    QC = "quality_control"             # Kontrola jakości
    ARCHIVE = "archive"                # Archiwizacja / koniec procesu


class PartStatus(str, Enum):
    """Status części w procesie."""
    PENDING = "pending"               # Czeka na przetworzenie
    IN_PROGRESS = "in_progress"       # Trakcie przetwarzania
    WAITING_INPUT = "waiting_input"   # Czeka na input (np. dane operatora)
    SUCCESS = "success"               # Pomyślnie przetworzono
    FAILED = "failed"                 # Błąd
    DEFECT = "defect"                 # Defekt - wymaga naprawy / odrzucenia


class QCResult(str, Enum):
    """Wynik kontroli jakości."""
    PASS = "pass"                     # Część OK
    FAIL_MINOR = "fail_minor"         # Drobne odchylenia (naprawialne)
    FAIL_MAJOR = "fail_major"         # Poważne błędy (odrzucenie)
    REWORK = "rework"                 # Wymaga przerobienia


# ============================================================================
# 2. DATACLASY
# ============================================================================

@dataclass(frozen=True)
class ProcessResult:
    """Wynik etapu procesu."""
    success: bool
    stage: ProcessStage
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PartJob:
    """Opis części w systemie produkcyjnym."""
    job_id: str                        # Unikalny ID zadania
    part_id: str                       # ID części metalowej
    current_stage: ProcessStage        # Aktualny etap
    status: PartStatus                 # Status
    
    # Dane z identyfikacji
    identified_model: Optional[str] = None
    model_confidence: Optional[float] = None
    material: Optional[str] = None
    category: Optional[str] = None
    
    # Dane z routingu
    machining_program_id: Optional[str] = None
    assigned_machine: Optional[str] = None
    quality_params: Optional[Dict] = None
    
    # Dane z QC
    qc_result: Optional[QCResult] = None
    qc_defects: List[str] = field(default_factory=list)
    
    # Śledzenie czasowe
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Logia
    stage_history: List[Tuple[ProcessStage, PartStatus, str]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class StageTransition:
    """Przejście między etapami."""
    job_id: str
    from_stage: ProcessStage
    to_stage: ProcessStage
    from_status: PartStatus
    to_status: PartStatus
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# 3. ROUTING ENGINE — Warunkowe wybory ścieżek
# ============================================================================

class RoutingEngine:
    """Silnik routingu: gdzie wysłać część después identyfikacji?"""
    
    def __init__(self):
        self.routing_rules = self._build_rules()
    
    def _build_rules(self) -> Dict[str, Dict]:
        """Zdefiniuj reguły routingu."""
        return {
            "fasteners": {
                "default_machine": "CNC_MILL_001",
                "program_pattern": "fastener_{material}_*.nc",
                "qc_checks": ["diameter", "thread", "surface"],
                "priority": 1,
            },
            "bearings": {
                "default_machine": "LATHE_002",
                "program_pattern": "bearing_{type}_*.nc",
                "qc_checks": ["roundness", "runout", "preload"],
                "priority": 2,
            },
            "shafts": {
                "default_machine": "CNC_LATHE_003",
                "program_pattern": "shaft_{diameter}mm_*.nc",
                "qc_checks": ["length", "diameter", "runout", "surface"],
                "priority": 2,
            },
            "plates": {
                "default_machine": "VERTICAL_MILL_004",
                "program_pattern": "plate_{size}_*.nc",
                "qc_checks": ["flatness", "thickness_uniformity"],
                "priority": 3,
            },
        }
    
    def route_by_category(self, category: str, material: str) -> Dict:
        """Wyznacz maszynę i program na podstawie kategorii i materiału."""
        rule = self.routing_rules.get(category, self.routing_rules["fasteners"])
        
        return {
            "machine": rule["default_machine"],
            "program_pattern": rule["program_pattern"],
            "qc_checks": rule["qc_checks"],
            "priority": rule["priority"],
            "material_factor": self._material_complexity(material),
        }
    
    def _material_complexity(self, material: str) -> float:
        """Zwróć współczynnik złożoności materiału."""
        complexities = {
            "Stal zwykła": 1.0,
            "Stal nierdzewna": 1.3,
            "Aluminium": 0.8,
            "Tytan": 1.5,
            "Brąz": 1.2,
        }
        return complexities.get(material, 1.0)


# ============================================================================
# 4. ETAPY PROCESU — Executory
# ============================================================================

class StageExecutor:
    """Bazowa klasa dla executorów etapów."""
    
    def __init__(self, stage: ProcessStage, name: str):
        self.stage = stage
        self.name = name
        self.logger = logging.getLogger(f"StageExecutor.{name}")
    
    async def execute(self, job: PartJob, context: Dict) -> ProcessResult:
        """Wykonaj etap. Nadpisz w podklasach."""
        raise NotImplementedError


class InputStageExecutor(StageExecutor):
    """Etap INPUT: przyjęcie części."""
    
    def __init__(self):
        super().__init__(ProcessStage.INPUT, "InputStage")
    
    async def execute(self, job: PartJob, context: Dict) -> ProcessResult:
        """
        Wejście: część fizycznie pojawia się w systemie.
        - Verificy całość obrazu/skanu
        - Zaloguj start
        """
        try:
            job.logs.append(f"[INPUT] Część {job.part_id} przyjęta do systemu")
            job.started_at = datetime.now().isoformat()
            
            return ProcessResult(
                success=True,
                stage=self.stage,
                data={"input_ready": True},
                metadata={"operator": context.get("operator", "system")},
            )
        except Exception as e:
            return ProcessResult(
                success=False,
                stage=self.stage,
                error=str(e),
            )


class ScanStageExecutor(StageExecutor):
    """Etap SCAN_3D: skanowanie części skanera 3D."""
    
    def __init__(self, comparator=None):
        super().__init__(ProcessStage.SCAN_3D, "ScanStage")
        self.comparator = comparator
    
    async def execute(self, job: PartJob, context: Dict) -> ProcessResult:
        """
        Skanowanie 3D: uzyskaj chmurę punktów.
        - Sprawdź jakość skanu
        - Przetwórz dane
        """
        try:
            scan_points = context.get("scan_points")
            if scan_points is None:
                return ProcessResult(
                    success=False,
                    stage=self.stage,
                    error="Brak danych skanera 3D",
                )
            
            # Ocena jakości skanu (placeholder)
            scan_quality = min(1.0, len(scan_points) / 5000)
            
            if scan_quality < 0.6:
                return ProcessResult(
                    success=False,
                    stage=self.stage,
                    error="Niska jakość skanu",
                    warnings=[f"Jakość skanu: {scan_quality:.1%}"],
                    metadata={"recommended_action": "rescan"},
                )
            
            job.logs.append(f"[SCAN] Skan 3D: {len(scan_points)} punktów, jakość: {scan_quality:.1%}")
            
            return ProcessResult(
                success=True,
                stage=self.stage,
                data={
                    "scan_points": scan_points,
                    "scan_quality": scan_quality,
                    "points_count": len(scan_points),
                },
            )
        except Exception as e:
            return ProcessResult(
                success=False,
                stage=self.stage,
                error=str(e),
            )


class IdentifyStageExecutor(StageExecutor):
    """Etap IDENTIFY: identyfikacja modelu."""
    
    def __init__(self, comparator=None):
        super().__init__(ProcessStage.IDENTIFY, "IdentifyStage")
        self.comparator = comparator
    
    async def execute(self, job: PartJob, context: Dict) -> ProcessResult:
        """
        Identyfikacja: porównaj skan z bazą modeli.
        - Uruchom 3D comparator
        - Przefiltruj wyniki po confidence
        """
        try:
            scan_points = context.get("scan_points")
            if self.comparator is None or scan_points is None:
                return ProcessResult(
                    success=False,
                    stage=self.stage,
                    error="Brak comparatora lub danych skanera",
                )
            
            # Identyfikacja
            results = self.comparator.identify(scan_points, top_k=1)
            
            if not results:
                return ProcessResult(
                    success=False,
                    stage=self.stage,
                    error="Żaden model nie pasuje do skanu",
                    metadata={"recommended_action": "manual_review"},
                )
            
            best_match = results[0]
            model_id = best_match["model_id"]
            confidence = best_match["final_score"]
            
            if confidence < 0.65:
                return ProcessResult(
                    success=False,
                    stage=self.stage,
                    error=f"Niska pewność identyfikacji: {confidence:.1%}",
                    warnings=[f"Najlepsze dopasowanie: {model_id} ({confidence:.1%})"],
                    metadata={"recommended_action": "manual_review"},
                )
            
            job.identified_model = model_id
            job.model_confidence = confidence
            job.logs.append(f"[IDENTIFY] Model: {model_id}, pewność: {confidence:.1%}")
            
            return ProcessResult(
                success=True,
                stage=self.stage,
                data={
                    "identified_model": model_id,
                    "confidence": confidence,
                    "all_candidates": results[:3],
                },
            )
        except Exception as e:
            return ProcessResult(
                success=False,
                stage=self.stage,
                error=str(e),
            )


class RouteStageExecutor(StageExecutor):
    """Etap ROUTE: routing do maszyny i programu."""
    
    def __init__(self, routing_engine: Optional[RoutingEngine] = None):
        super().__init__(ProcessStage.ROUTE, "RouteStage")
        self.routing_engine = routing_engine or RoutingEngine()
    
    async def execute(self, job: PartJob, context: Dict) -> ProcessResult:
        """
        Routing: gdzie wysłać część?
        - Ustal maszynę na podstawie kategorii/materiału
        - Zaloguj routing
        """
        try:
            category = context.get("category", "fasteners")
            material = context.get("material", "Stal zwykła")
            
            routing_decision = self.routing_engine.route_by_category(category, material)
            
            job.assigned_machine = routing_decision["machine"]
            job.quality_params = {
                "checks": routing_decision["qc_checks"],
                "priority": routing_decision["priority"],
            }
            job.logs.append(
                f"[ROUTE] Przydzielono: {routing_decision['machine']}, "
                f"program: {routing_decision['program_pattern']}"
            )
            
            return ProcessResult(
                success=True,
                stage=self.stage,
                data=routing_decision,
            )
        except Exception as e:
            return ProcessResult(
                success=False,
                stage=self.stage,
                error=str(e),
            )


class MachineStageExecutor(StageExecutor):
    """Etap MACHINE: obróbka na CNC."""
    
    def __init__(self):
        super().__init__(ProcessStage.MACHINE, "MachineStage")
    
    async def execute(self, job: PartJob, context: Dict) -> ProcessResult:
        """
        Obróbka CNC (simulator).
        - Załóż sukces jako placeholder
        - Zaloguj parametry
        """
        try:
            machine = context.get("machine", "CNC_MILL_001")
            program = context.get("program", "default.nc")
            
            # Symulacja: sukces
            job.logs.append(f"[MACHINE] Obróbka: {machine}, program: {program}")
            
            return ProcessResult(
                success=True,
                stage=self.stage,
                data={
                    "machine": machine,
                    "program": program,
                    "cycle_time_seconds": 120,
                    "status": "completed",
                },
            )
        except Exception as e:
            return ProcessResult(
                success=False,
                stage=self.stage,
                error=str(e),
            )


class QCStageExecutor(StageExecutor):
    """Etap QC: kontrola jakości."""
    
    def __init__(self, verifier=None):
        super().__init__(ProcessStage.QC, "QCStage")
        self.verifier = verifier
    
    async def execute(self, job: PartJob, context: Dict) -> ProcessResult:
        """
        Kontrola jakości: sprawdź wymiary, powierzchnię, itp.
        - Symulacja: losowy wynik QC
        """
        try:
            qc_checks = context.get("qc_checks", ["dimension", "surface"])
            
            # Symulacja: zwróć PASS z 80% prawdopodobieństwem
            import random
            results = {
                "dimension": random.uniform(0.95, 1.05),
                "surface": random.uniform(1.0, 1.6),
                "roundness": random.uniform(0.0, 0.05),
            }
            
            # Decyzja QC
            qc_result = QCResult.PASS
            defects = []
            
            if any(v > 1.02 for v in results.values()):
                qc_result = QCResult.FAIL_MINOR
                defects.append("Wymiary poza tolerancją (minor)")
            
            job.qc_result = qc_result
            job.qc_defects = defects
            job.logs.append(f"[QC] Wynik: {qc_result.value}, defekty: {defects}")
            
            return ProcessResult(
                success=qc_result == QCResult.PASS,
                stage=self.stage,
                data={
                    "qc_result": qc_result.value,
                    "measurements": results,
                    "defects": defects,
                },
            )
        except Exception as e:
            return ProcessResult(
                success=False,
                stage=self.stage,
                error=str(e),
            )


class ArchiveStageExecutor(StageExecutor):
    """Etap ARCHIVE: archiwizacja i koniec."""
    
    def __init__(self, repo=None):
        super().__init__(ProcessStage.ARCHIVE, "ArchiveStage")
        self.repo = repo
    
    async def execute(self, job: PartJob, context: Dict) -> ProcessResult:
        """
        Archiwizacja: zapisz wynik do repozytorium.
        - Zaktualizuj status
        - Zaloguj koniec
        """
        try:
            qc_result = context.get("qc_result", "unknown")
            
            job.completed_at = datetime.now().isoformat()
            job.logs.append(f"[ARCHIVE] Część zarchiwizowana, QC: {qc_result}")
            
            return ProcessResult(
                success=True,
                stage=self.stage,
                data={
                    "archived": True,
                    "final_status": qc_result,
                },
            )
        except Exception as e:
            return ProcessResult(
                success=False,
                stage=self.stage,
                error=str(e),
            )


# ============================================================================
# 5. WORKFLOW ENGINE — Orkestracja
# ============================================================================

class ProductionWorkflowEngine:
    """Główny silnik orkiestrujący proces produkcyjny."""
    
    def __init__(self):
        self.stages = {
            ProcessStage.INPUT: InputStageExecutor(),
            ProcessStage.SCAN_3D: ScanStageExecutor(),
            ProcessStage.IDENTIFY: IdentifyStageExecutor(),
            ProcessStage.ROUTE: RouteStageExecutor(),
            ProcessStage.MACHINE: MachineStageExecutor(),
            ProcessStage.QC: QCStageExecutor(),
            ProcessStage.ARCHIVE: ArchiveStageExecutor(),
        }
        self.transitions: List[StageTransition] = []
    
    async def process_part(
        self,
        job: PartJob,
        context: Dict,
    ) -> Tuple[PartJob, List[ProcessResult]]:
        """
        Główny pipeline przetwarzania części.
        Podąża wzdłuż etapów i rejestruje przejścia.
        """
        results = []
        stage_order = [
            ProcessStage.INPUT,
            ProcessStage.SCAN_3D,
            ProcessStage.IDENTIFY,
            ProcessStage.ROUTE,
            ProcessStage.MACHINE,
            ProcessStage.QC,
            ProcessStage.ARCHIVE,
        ]
        
        for stage in stage_order:
            executor = self.stages[stage]
            job.current_stage = stage
            job.status = PartStatus.IN_PROGRESS
            
            # Uruchom etap
            result = await executor.execute(job, context)
            results.append(result)
            
            # Zaloguj przejście
            old_status = job.status
            if result.success:
                job.status = PartStatus.SUCCESS
            else:
                job.status = PartStatus.FAILED
                if result.metadata.get("recommended_action") == "manual_review":
                    job.status = PartStatus.WAITING_INPUT
                break
            
            transition = StageTransition(
                job_id=job.job_id,
                from_stage=stage,
                to_stage=stage_order[stage_order.index(stage) + 1] if stage != stage_order[-1] else stage,
                from_status=old_status,
                to_status=job.status,
                reason=f"Etap {stage.value} zakończony",
            )
            self.transitions.append(transition)
            
            # Early exit na błąd
            if not result.success:
                break
        
        job.updated_at = datetime.now().isoformat()
        return job, results
    
    def get_workflow_summary(self) -> Dict:
        """Podsumowanie statystyk workflow."""
        return {
            "total_transitions": len(self.transitions),
            "stages_configured": len(self.stages),
            "transitions_by_stage": self._count_by_stage(),
        }
    
    def _count_by_stage(self) -> Dict[str, int]:
        """Policz przejścia po etapach."""
        counts = {}
        for t in self.transitions:
            key = t.from_stage.value
            counts[key] = counts.get(key, 0) + 1
        return counts
