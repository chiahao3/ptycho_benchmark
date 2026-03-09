"""
System and environment diagnostic reporting.

This module provides utilities to query and log the current hardware 
(CPU, Memory, GPU) and software (OS, Python, dependencies) environment. 
It includes specific support for detecting SLURM cluster allocations and 
identifying NVIDIA Multi-Instance GPU (MIG) configurations.
"""

import subprocess

def is_mig_enabled():
    """Detects if any NVIDIA GPU on the system is operating in MIG mode.
    

    Multi-Instance GPU (MIG) allows a physical GPU to be securely partitioned 
    into multiple separate GPU instances. This function queries `nvidia-smi` 
    to check if this hardware partitioning is currently active, which is 
    important because certain multi-GPU communication backends (like NCCL) 
    do not fully support MIG slices.

    Returns:
        bool: True if MIG mode is enabled on any detected GPU, False if it 
        is disabled, or if the detection fails (e.g., `nvidia-smi` not found).
    """
    try:
        # Run the `nvidia-smi` command to query MIG mode
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=mig.mode.current", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Check for errors in the command execution
        if result.returncode != 0:
            print(f"Error running nvidia-smi: {result.stderr.strip()}")
            return False
        
        # Parse the output to check for MIG mode
        mig_modes = result.stdout.strip().split("\n")
        for mode in mig_modes:
            if mode.strip() == "Enabled":
                return True
        
        return False
    except FileNotFoundError:
        # `nvidia-smi` is not available
        print("nvidia-smi not found. Unable to detect MIG mode.")
        return False
    except Exception as e:
        # Catch other unexpected errors
        print(f"Error detecting MIG mode: {e}")
        return False

def print_system_info():
    """Logs comprehensive system hardware and operating system information.

    This function records the OS platform, processor architecture, available 
    CPU cores, and system memory. It automatically detects if the code is 
    running inside a SLURM job allocation and reports the SLURM-restricted 
    resources instead of the total physical node resources. It subsequently 
    triggers GPU and package diagnostics.
    """
    import os
    import platform
    import sys
    
    print("### System information ###")
    
    # Operating system information
    print(f"Platform: {platform.platform()}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"OS Version: {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # CPU cores
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        cpus =  int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    else:
        # Fallback to the total number of CPU cores on the node
        cpus = os.cpu_count()
    print(f"Available CPU cores: {cpus}")
    
    # Memory information
    if 'SLURM_MEM_PER_NODE' in os.environ:
        # Memory allocated per node by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_NODE']) / 1024  # Convert MB to GB
        print(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    elif 'SLURM_MEM_PER_CPU' in os.environ:
        # Memory allocated per CPU by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_CPU']) * cpus / 1024  # Convert MB to GB
        print(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    else:
        try:
            import psutil
            # Fallback to system memory information
            mem = psutil.virtual_memory()
            print(f"Total Memory: {mem.total / (1024 ** 3):.2f} GB")
            print(f"Available Memory: {mem.available / (1024 ** 3):.2f} GB")
        except ImportError:
            print("Memory information will be available after `conda install conda-forge::psutil`")
    print(" ")
            
    # GPU information
    print_gpu_info()
    print(" ")
    
    # Python version and executable
    print("### Python information ###")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    print(" ")
    
    # Packages information (numpy, PyTorch, Optuna, Accelerate, PtyRAD)
    print_packages_info()
    print(" ")

def print_gpu_info():
    """Logs physical GPU hardware and compute details for PyTorch and JAX.

    Detects and reports available compute backends, including NVIDIA CUDA, 
    Apple Silicon MPS, and JAX GPU/TPU backends. It provides actionable 
    troubleshooting tips if a GPU is expected but cannot be found.
    """
    print(f"### GPU information ###")
    
    # ==========================================
    # 1. PyTorch Detection
    # ==========================================

    try:
        import torch
        print("--- PyTorch Backend ---")
        
        if torch.backends.cuda.is_built() and torch.cuda.is_available():
            print(f"CUDA Available: {torch.cuda.is_available()}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Available CUDA GPUs: {[torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())]}")
            print(f"CUDA Compute Capability: {[f'{major}.{minor}' for (major, minor) in [torch.cuda.get_device_capability(d) for d in range(torch.cuda.device_count())]]}")
            print("  INFO: For torch.compile with Triton, you'll need CUDA GPU with Compute Capability >= 7.0.")
            print("        In addition, Triton does not directly support Windows.")
            print("        For Windows users, please follow the instruction and download `triton-windows` from https://github.com/woct0rdho/triton-windows.")
            print(f"MIG (Multi-Instance GPU) mode = {is_mig_enabled()}")
            print("  INFO: MIG splits a physical GPU into multiple GPU slices, but multiGPU does not support these MIG slices.")
            print("        In addition, multiGPU is currently only available on Linux due to the limited NCCL support.")
            print("      -> If you're doing normal reconstruction/hypertune, you can safely ignore this.")
            print("      -> If you want to do multiGPU, you must provide multiple 'full' GPUs that are not in MIG mode.")
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            print(f"MPS Available: {torch.backends.mps.is_available()}")
        elif torch.backends.cuda.is_built() or torch.backends.mps.is_built():
            print("WARNING: GPU support built with PyTorch, but could not find any existing / compatible GPU device.")
            print("         PtyRAD will fall back to CPU which is much slower in performance")
            print("         -> If you're using a CPU-only machine, you can safely ignore this.")
            print("         -> If you believe you *do* have a GPU, please check the compatibility:")
            print("           - Are the correct NVIDIA drivers installed?")
            print("           - Is your CUDA runtime version compatible with PyTorch?")
            print("           Tips: Run `nvidia-smi` in your terminal for NVIDIA driver and CUDA runtime information.")
            print("           Tips: Run `conda list torch` in your terminal (with `ptyrad` environment activated) to check the installed PyTorch version.")
        else:
            print("WARNING: No GPU backend (CUDA or MPS) built into this PyTorch install.")
            print("         PtyRAD will fall back to CPU which is much slower in performance")
            print("         Please consider reinstalling PyTorch with GPU support if available.")
            print("         See https://github.com/chiahao3/ptyrad for PtyRAD installation guide.")
    except ImportError:
        print("--- PyTorch Backend ---")
        print("WARNING: No GPU information because PyTorch can't be imported.")
        print("         Please install PyTorch because it's the crucial dependency of PtyRAD.")
        print("         See https://github.com/chiahao3/ptyrad for PtyRAD installation guide.")

    # ==========================================
    # 2. JAX Detection (Phaser, etc.)
    # ==========================================
    try:
        import jax
        from jax.extend.backend import backends
        print("--- JAX Backend ---")
        available_backends = backends()
        print(f"Available JAX Backends: {available_backends.keys()}")
        
        if 'cuda' in available_backends.keys():
            devices = jax.devices('gpu')
            print(f"Available JAX GPUs: {[d.device_kind for d in devices]}")
        elif 'tpu' in available_backends.keys():
            devices = jax.devices('tpu')
            print(f"Available JAX TPUs: {len(devices)} cores detected.")
        else:
            print(f"WARNING: JAX is installed but could not find any GPU/TPU.")
            print("         If you expect a GPU/TPU, ensure you installed jaxlib with CUDA support:")
            print("         pip install -U \"jax[cuda12]\"")
            
    except ImportError:
        print("--- JAX Backend ---")
        print("Not installed in this environment.")
    
def print_packages_info():
    """Logs installed versions of critical Python dependencies.

    Reports the environment versions of Numpy, PyTorch, Optuna, and Accelerate. 
    Crucially, it checks the runtime version of the `ptyrad` package against 
    the installation metadata to detect "stale metadata" scenarios common in 
    editable (`pip install -e .`) installs, warning the user if a mismatch 
    is found.
    """
    
    import importlib
    import importlib.metadata
    print("### Packages information ###")

    # Print package versions
    basic_packages = [
        ("Numpy", "numpy"),
        ("Cupy", "cupy"),
        ("Jax", "jax"),
        ("PyTorch", "torch"),
        ("Optuna", "optuna"),
        ("Accelerate", "accelerate"),
    ]

    # Check versions for relevant packages
    for display_name, module_name in basic_packages:
        try:
            # Try to get the version from package metadata (installed version)
            version = importlib.metadata.version(module_name)
            print(f"{display_name} Version (metadata): {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{display_name} not found in the environment.")
        except Exception as e:
            print(f"Error retrieving version for {display_name}: {e}")

    # # Check the version and path of the used PtyRAD package
    # # In general:
    # # - `ptyrad.__version__` reflects the actual code you're running (from source files).
    # # - `importlib.metadata.version("ptyrad")` reflects the version during install.
    # # 
    # # Note that we're focusing on the version/path of the actual imported PtyRAD.
    # # If there are both an installed version of PtyRAD in the environment (site-packages/) and a local copy in the working directory (src/ptyrad),
    # # Python will prioritize the version in the working directory.
    # #
    # # When using `pip install -e .`, only the version metadata gets recorded, which won't be updated until you reinstall.
    # # As a result, a user who pulls new code from the repo will have their `__init__.py` updated, but the version metadata recorded by pip will remain unchanged.
    # # Therefore, it is better to retrieve the version directly from `module.__version__` for now, as this will reflect the actual local version being used.
    # # In a release install (pip or conda), metadata and __version__ will match due to the dynamic version in pyproject.toml
    # # During editable installs, metadata may lag behind source changes.

    ptycho_packages = [
        ("PtyRAD", "ptyrad"),
        ("phaser", "phaser"),
        ("py4DSTEM", "py4DSTEM"),
        ("quantem", "quantem"),
    ]

    for display_name, module_name in ptycho_packages:
        try:
            # Import ptyrad (which will prioritize the local version if available)
            module = importlib.import_module(f'{module_name}')
            runtime_version = module.__version__
            metadata_version = importlib.metadata.version(f"{module_name}")
            print(f"{display_name} Version ({module_name}/__init__.py): {runtime_version}")
            print(f"{display_name} is located at: {module.__file__}") # For editable install this will be in package src/, while full install would make a copy at site-packages/
            
            if runtime_version and metadata_version and runtime_version != metadata_version:
                print("WARNING: Version mismatch detected!")
                print(f"  Runtime version : {runtime_version} (retrieved from current source file: {module_name}/__init__.py)")
                print(f"  Metadata version: {metadata_version} (recorded during previous `pip/conda install`)")
                print("  This likely means you downloaded new codes from repo but forgot to update the installed metadata.")
                print("  This does not affect the code execution because the runtime version of code is always used, but this can lead to misleading version logs.")
                print("  To fix this, re-run: `pip install -e . --no-deps` at the package root directory.")
            
        except ImportError:
            print(f"{display_name} not found locally")
        except AttributeError:
            print(f"{display_name} imported, but no __version__ attribute found.")
        except Exception as e:
            print(f"Error retrieving version for {display_name}: {e}")