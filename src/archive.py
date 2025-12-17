"""
GCS Archiving for Experiment Results.

Archives experiment data to GCS bucket and manages local cleanup.
"""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


# GCS bucket for experiment archives
GCS_ARCHIVE_BUCKET = "brightfoxai-documents"
GCS_ARCHIVE_PATH = "BRIGHTFOXAI/EVAL_ARCHIVE"


@dataclass
class ArchiveManifest:
    """Manifest of archived experiment data."""
    experiment_name: str
    archive_timestamp: str
    gcs_path: str
    files_archived: List[str]
    total_size_bytes: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "archive_timestamp": self.archive_timestamp,
            "gcs_path": self.gcs_path,
            "files_archived": self.files_archived,
            "total_size_bytes": self.total_size_bytes,
            "metadata": self.metadata,
        }


def archive_experiment_to_gcs(
    experiment_dir: Path,
    experiment_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    bucket: str = GCS_ARCHIVE_BUCKET,
    archive_path: str = GCS_ARCHIVE_PATH,
) -> ArchiveManifest:
    """
    Archive experiment directory to GCS bucket.
    
    Args:
        experiment_dir: Local directory containing experiment files
        experiment_name: Name of the experiment
        metadata: Optional metadata to include in manifest
        bucket: GCS bucket name
        archive_path: Path within bucket for archives
        
    Returns:
        ArchiveManifest with archive details
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_dest = f"gs://{bucket}/{archive_path}/{experiment_name}_{timestamp}"
    
    # Get list of files to archive
    files_to_archive = []
    total_size = 0
    
    for f in experiment_dir.rglob("*"):
        if f.is_file():
            files_to_archive.append(str(f.relative_to(experiment_dir)))
            total_size += f.stat().st_size
    
    print(f"Archiving {len(files_to_archive)} files ({total_size / 1024 / 1024:.1f} MB) to {gcs_dest}")
    
    # Upload directory to GCS
    cmd = f'gsutil -m cp -r "{experiment_dir}/*" "{gcs_dest}/"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"GCS upload failed: {result.stderr}")
    
    # Create manifest
    manifest = ArchiveManifest(
        experiment_name=experiment_name,
        archive_timestamp=datetime.now().isoformat(),
        gcs_path=gcs_dest,
        files_archived=files_to_archive,
        total_size_bytes=total_size,
        metadata=metadata or {},
    )
    
    # Upload manifest
    manifest_path = experiment_dir / "archive_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    
    manifest_gcs = f"{gcs_dest}/archive_manifest.json"
    subprocess.run(f'gsutil cp "{manifest_path}" "{manifest_gcs}"', shell=True, check=True)
    
    print(f"✅ Archived to: {gcs_dest}")
    
    return manifest


def archive_experiment_results(
    experiment_name: str,
    results_file: Path,
    checkpoint_file: Optional[Path] = None,
    report_file: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
    bucket: str = GCS_ARCHIVE_BUCKET,
    archive_path: str = GCS_ARCHIVE_PATH,
) -> str:
    """
    Archive specific experiment result files to GCS.
    
    Args:
        experiment_name: Name of the experiment
        results_file: Path to results JSON file
        checkpoint_file: Optional path to checkpoint JSONL file
        report_file: Optional path to report markdown file
        metadata: Optional metadata to include
        bucket: GCS bucket name
        archive_path: Path within bucket
        
    Returns:
        GCS path where files were archived
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_dest = f"gs://{bucket}/{archive_path}/{experiment_name}_{timestamp}"
    
    files_uploaded = []
    
    # Upload results file
    if results_file.exists():
        gcs_file = f"{gcs_dest}/{results_file.name}"
        subprocess.run(f'gsutil cp "{results_file}" "{gcs_file}"', shell=True, check=True)
        files_uploaded.append(results_file.name)
        print(f"  ✓ {results_file.name}")
    
    # Upload checkpoint file
    if checkpoint_file and checkpoint_file.exists():
        gcs_file = f"{gcs_dest}/{checkpoint_file.name}"
        subprocess.run(f'gsutil cp "{checkpoint_file}" "{gcs_file}"', shell=True, check=True)
        files_uploaded.append(checkpoint_file.name)
        print(f"  ✓ {checkpoint_file.name}")
    
    # Upload report file
    if report_file and report_file.exists():
        gcs_file = f"{gcs_dest}/{report_file.name}"
        subprocess.run(f'gsutil cp "{report_file}" "{gcs_file}"', shell=True, check=True)
        files_uploaded.append(report_file.name)
        print(f"  ✓ {report_file.name}")
    
    # Create and upload manifest
    manifest = {
        "experiment_name": experiment_name,
        "archive_timestamp": datetime.now().isoformat(),
        "gcs_path": gcs_dest,
        "files_archived": files_uploaded,
        "metadata": metadata or {},
    }
    
    manifest_json = json.dumps(manifest, indent=2)
    manifest_cmd = f'echo \'{manifest_json}\' | gsutil cp - "{gcs_dest}/manifest.json"'
    subprocess.run(manifest_cmd, shell=True, check=True)
    
    print(f"✅ Archived {len(files_uploaded)} files to: {gcs_dest}")
    
    return gcs_dest


def list_archived_experiments(
    bucket: str = GCS_ARCHIVE_BUCKET,
    archive_path: str = GCS_ARCHIVE_PATH,
) -> List[str]:
    """List all archived experiments in GCS."""
    gcs_path = f"gs://{bucket}/{archive_path}/"
    
    result = subprocess.run(
        f'gsutil ls "{gcs_path}"',
        shell=True, capture_output=True, text=True
    )
    
    if result.returncode != 0:
        return []
    
    experiments = []
    for line in result.stdout.strip().split("\n"):
        if line:
            # Extract experiment name from path
            name = line.rstrip("/").split("/")[-1]
            experiments.append(name)
    
    return experiments


def download_archived_experiment(
    experiment_name: str,
    local_dir: Path,
    bucket: str = GCS_ARCHIVE_BUCKET,
    archive_path: str = GCS_ARCHIVE_PATH,
) -> bool:
    """Download an archived experiment from GCS."""
    gcs_path = f"gs://{bucket}/{archive_path}/{experiment_name}"
    
    local_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = f'gsutil -m cp -r "{gcs_path}/*" "{local_dir}/"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Download failed: {result.stderr}")
        return False
    
    print(f"✅ Downloaded to: {local_dir}")
    return True


# Files that should be gitignored (large data files)
LARGE_FILE_PATTERNS = [
    "retrieval_cache.json",
    "*_checkpoint.jsonl",
    "all_chunks.json",
    "*.jsonl",
]


def get_gitignore_entries() -> List[str]:
    """Get recommended .gitignore entries for large experiment files."""
    return [
        "# Large experiment data files (archived to GCS)",
        "bfai_eval_suite/experiments/**/retrieval_cache.json",
        "bfai_eval_suite/experiments/**/*_checkpoint.jsonl",
        "bfai_eval_suite/data/all_chunks.json",
        "bfai_eval_suite/data/chunks/*.jsonl",
    ]
