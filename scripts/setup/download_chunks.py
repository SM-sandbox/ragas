"""Download chunk data from GCS bucket"""
import os
import json
import subprocess
from pathlib import Path

# Add parent to path for config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config


def list_documents():
    """List all document folders in the GCS bucket"""
    cmd = f"gsutil ls gs://{config.GCS_BUCKET}/{config.GCS_CHUNKS_PATH}/"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Failed to list GCS bucket: {result.stderr}")
    
    folders = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    return folders


def download_chunks(output_dir: str = None):
    """Download all chunk JSONL files from GCS"""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / config.DATA_DIR / "chunks"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    folders = list_documents()
    print(f"Found {len(folders)} document folders")
    
    all_chunks = []
    
    for folder in folders:
        # Extract document name from folder path
        doc_name = folder.rstrip('/').split('/')[-1]
        
        # Find the chunks.jsonl file
        chunks_path = f"{folder}{doc_name}_chunks.jsonl"
        local_file = output_dir / f"{doc_name}_chunks.jsonl"
        
        # Download the file
        cmd = f'gsutil cp "{chunks_path}" "{local_file}" 2>/dev/null'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0 and local_file.exists():
            # Read and collect chunks
            with open(local_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            chunk['source_document'] = doc_name
                            all_chunks.append(chunk)
                        except json.JSONDecodeError:
                            continue
            print(f"✓ Downloaded: {doc_name} ({len(all_chunks)} total chunks)")
        else:
            # Try alternative naming patterns
            alt_patterns = [
                f"{folder}chunks.jsonl",
                f"{folder}{doc_name.replace(' ', '_')}_chunks.jsonl",
            ]
            for alt_path in alt_patterns:
                cmd = f'gsutil cp "{alt_path}" "{local_file}" 2>/dev/null'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0 and local_file.exists():
                    with open(local_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    chunk = json.loads(line)
                                    chunk['source_document'] = doc_name
                                    all_chunks.append(chunk)
                                except json.JSONDecodeError:
                                    continue
                    print(f"✓ Downloaded (alt): {doc_name}")
                    break
    
    # Save combined chunks file
    combined_file = output_dir.parent / "all_chunks.json"
    with open(combined_file, 'w') as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"\n✓ Total chunks downloaded: {len(all_chunks)}")
    print(f"✓ Combined file saved to: {combined_file}")
    
    return all_chunks


if __name__ == "__main__":
    download_chunks()
