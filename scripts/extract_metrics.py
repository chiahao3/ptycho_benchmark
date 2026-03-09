import re
import json
import argparse
from pathlib import Path

# ==========================================
# CONFIGURATION: PACKAGE PARSING RULES
# ==========================================
PACKAGE_CONFIGS = {
    "quantem": {
        "marker": re.compile(r"### Running \((.*?)\) = \((.*?)\) ###"),
        "iteration": re.compile(r"Iter \d+ took ([0-9.]+) sec"),
        "success": re.compile(r"Completed \d+ iters"),
        "error": re.compile(r"error|CUDA out of memory|OOM", re.IGNORECASE),
    },
    "ptyrad": {
        "marker": re.compile(r"### Running \((.*?)\)\s*=\s*\((.*?)\) ###"),
        "iteration": re.compile(r"Iter:\s*\d+,.*?in\s+((?:\d+\s*min\s+)?[0-9.]+)\s*sec"),
        "success": re.compile(r"Finished \d+ iterations"),
        "error": re.compile(r"error|CUDA out of memory|OOM", re.IGNORECASE),
    },
    "phaser": {
        "marker": re.compile(r"### Running \((.*?)\)\s*=\s*\((.*?)\) ###"),
        "iteration": re.compile(r"Finished iter\s+\d+/\d+\s+\[(\d{2}:\d{2}\.[0-9]+)\]"),
        "success": re.compile(r"Engine finished!|Reconstruction finished!"),
        "error": re.compile(r"RESOURCE_EXHAUSTED|Out of memory", re.IGNORECASE),
    },
}

PREFERRED_ORDER = [
    "date",
    "package", 
    "label", 
    "version",
    "backend",
    "device",
    "algorithm",
    "dataset",
    "Npix",
    "Nscans",
    "status", 
    "round_idx", 
    "batch", 
    "pmode", 
    "slice", 
    "iter_times",
]

def auto_cast(val_str):
    """Helper to convert string values to int or float for cleaner dataframes."""
    val_str = val_str.strip()
    try:
        if '.' in val_str:
            return float(val_str)
        return int(val_str)
    except ValueError:
        return val_str

def parse_time(time_str):
    """Converts MM:SS.mmm, X min Y, or raw seconds into a pure float."""
    time_str = time_str.strip()
    
    # Handle phaser format "[MM:SS.mmm]"
    if ':' in time_str:
        minutes, seconds = time_str.split(':')
        return (float(minutes) * 60) + float(seconds)
        
    # Handle ptyrad format "X min Y" and " Y sec"
    if 'min' in time_str:
        parts = time_str.split('min')
        minutes = float(parts[0].strip())
        seconds = float(parts[1].strip())
        return (minutes * 60) + seconds
        
    # Handle raw seconds "Y.YYY"
    return float(time_str)

def create_flat_record(status, current_params, current_times, metadata):
    """Merges all data into a single flat dictionary with a preferred key order."""
    # 1. Pool all the data together
    raw_data = {}
    if metadata:
        raw_data.update(metadata)
    raw_data["status"] = status
    raw_data.update(current_params)
    raw_data["iter_times"] = current_times

    # 2. Build the final ordered dictionary
    ordered_record = {}
    
    # Extract preferred keys first
    for key in PREFERRED_ORDER:
        if key in raw_data:
            ordered_record[key] = raw_data.pop(key)
            
    # Dump any remaining keys (e.g., extra metadata or new loop variables) at the end
    ordered_record.update(raw_data)
    
    return ordered_record

# ==========================================
# CORE PARSING & EXPORT LOGIC
# ==========================================

def parse_log(log_path, config, metadata=None):
    """
    Single-pass state machine to extract metrics from a log file.
    
    Args:
        log_path (str/Path): Path to the log file.
        config (dict): Regex configuration dictionary for the package.
        metadata (dict, optional): Static information (device, version, Npix, etc.) 
                                   to inject into every record.
    """
    records = []
    metadata = metadata or {} 
    
    hunting = True
    current_params = {}
    current_times = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # ---------------------------------------------------------
            # STATE: HUNTING (Looking for the next parameter block)
            # ---------------------------------------------------------    
            if hunting:
                match = config["marker"].search(line)
                if match:
                    keys_raw = match.group(1).split(',')
                    vals_raw = match.group(2).split(',')
                    current_params = {k.strip(): auto_cast(v) for k, v in zip(keys_raw, vals_raw)}
                    current_times = []
                    hunting = False
            
            # ---------------------------------------------------------
            # STATE: ACTIVE (Collecting times or waiting for termination)
            # ---------------------------------------------------------
            else:
                # 1. Check for iteration time
                iter_match = config["iteration"].search(line)
                if iter_match:
                    current_times.append(parse_time(iter_match.group(1)))
                    continue
                # 2. Check for successful completion
                if config["success"].search(line):
                    records.append(create_flat_record("success", current_params, current_times, metadata))
                    hunting = True 
                    continue
                # 3. Check for errors / OOM
                if config["error"].search(line):
                    records.append(create_flat_record("OOM", current_params, current_times, metadata))
                    hunting = True 
                    continue
                # 4. Implicit boundary catch (Safety net)
                # If we suddenly hit a new marker line while collecting, the 
                # previous block was interrupted without a clear error message.
                new_marker_match = config["marker"].search(line)
                if new_marker_match:
                    records.append(create_flat_record("interrupted", current_params, current_times, metadata))
                    
                    keys_raw = new_marker_match.group(1).split(',')
                    vals_raw = new_marker_match.group(2).split(',')
                    current_params = {k.strip(): auto_cast(v) for k, v in zip(keys_raw, vals_raw)}
                    current_times = []

    if not hunting and current_params:
         records.append(create_flat_record("incomplete_eof", current_params, current_times, metadata))
         
    return records

def export_to_ndjson(records, out_path, mode='a'):
    """Encapsulated export logic for ndjson files."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, mode, encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    return out_path

# ==========================================
# CLI ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Extract benchmark metrics from log files to ndjson.")
    parser.add_argument("--log", required=True, help="Path to the raw text log file.")
    parser.add_argument("--pkg", required=True, choices=PACKAGE_CONFIGS.keys(), help="Which package format to use.")
    parser.add_argument("--label", required=True, help="Labels (e.g., ptyrad_b13_pt2.10_a100).")
    parser.add_argument("--out", required=True, help="Output .ndjson file path.")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        return

    config = PACKAGE_CONFIGS[args.pkg]
    
    # Pack the CLI arguments into the new metadata dictionary format
    metadata = {
        "label": args.label,
        "package": args.pkg 
    }
    
    records = parse_log(log_path, config, metadata)
    export_to_ndjson(records, args.out)
            
    print(f"✅ Extracted {len(records)} blocks from {log_path.name} -> {args.out}")

if __name__ == "__main__":
    main()