# Python script to run PtyRAD
# Updated by Chia-Hao Lee on 2026.03.08

import argparse
import gc

import torch
from ptyrad.params import load_params
from ptyrad.runtime.device import set_gpu_device
from ptyrad.runtime.diagnostics import print_system_info
from ptyrad.runtime.logging import LoggingManager
from ptyrad.solver import PtyRADSolver

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run PtyRAD", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--skip_validate", action="store_true", help="Skip parameter validation and default filling. Use only if your params file is complete and consistent.")
    parser.add_argument("--gpuid", type=str, required=False, default="0", help="GPU ID to use ('acc', 'cpu', or an integer)")
    parser.add_argument("--jobid", type=int, required=False, default=0, help="Unique identifier for hypertune mode with multiple GPU workers")
    args = parser.parse_args()
    
    for round_idx in range(1, 6):
        for batch in [1024, 512, 256, 128, 64, 32, 16, 8, 4]:
            for pmode in [1,3,6,12]:
                for slice in [1,3,6,12]:
                    try:
                        # Setup LoggingManager
                        LoggingManager(
                            log_file='ptyrad_log.txt',
                            log_dir='auto',
                            prefix_time='datetime',
                            prefix_jobid=args.jobid,
                            append_to_file=True,
                            show_timestamp=True,
                            verbosity='INFO'
                        )

                        print_system_info()
                        params = load_params(args.params_path, validate=not args.skip_validate)
                        device = set_gpu_device(args.gpuid)
                                        
                        # Run ptyrad
                        print(f"### Running (round_idx, batch, pmode, slice) = {(round_idx, batch, pmode, slice)} ###")
                        
                        params['recon_params']['output_dir'] += f'_r{str(round_idx)}/'
                        params['recon_params']['BATCH_SIZE']['size'] = batch
                        params['init_params']['probe_pmode_max'] = pmode
                        params['init_params']['obj_Nlayer'] = slice
                        params['init_params']['obj_slice_thickness'] = round(12/slice, 2)
                        
                        ptycho_solver = PtyRADSolver(params, device=device)
                        ptycho_solver.run()
                        
                    except Exception as e:
                        print(f"An error occurred for (round, batch, pmode, slice) = {(round_idx, batch, pmode, slice)}: {e}")

                    finally:
                        del ptycho_solver  # Clear model
                        torch.cuda.empty_cache()
                        gc.collect()

