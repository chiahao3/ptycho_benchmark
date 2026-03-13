# Python script to run PtyRAD
# Updated by Chia-Hao Lee on 2026.03.11

import argparse
import gc
import itertools
import datetime

import torch
from ptyrad.params import PtyRADParams, InitParams
from ptyrad.runtime.device import set_gpu_device
from ptyrad.runtime.diagnostics import print_system_info
from ptyrad.runtime.logging import LoggingManager
from ptyrad.solver import PtyRADSolver

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
            description="Run PtyRAD", 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    
    # Generate today's date in YYYYMMDD format (e.g., "20260310")
    today_str = datetime.datetime.now().strftime("%Y%m%d")

    # Benchmark args
    parser.add_argument("--gpuid", type=int, default=0, help="GPU ID to use, default is 0")
    parser.add_argument("--device", type=str, required=True, help="Device string for output folder (e.g., RTX3080 or A100_20GB)")
    parser.add_argument("--date", type=str, default=today_str, help="Date string for output folder (e.g., 20260310)")
    parser.add_argument("--label", type=str, default='ptyrad', help="Arbitrary label string for identification")
    parser.add_argument("--round_idx", type=int, nargs='+', default=[1, 2, 3, 4, 5], help="List of round indices")
    parser.add_argument("--batches", type=int, nargs='+', default=[1024, 512, 256, 128, 64, 32, 16, 8, 4], help="List of batch sizes")
    parser.add_argument("--pmodes", type=int, nargs='+', default=[1, 3, 6, 12], help="List of probe modes")
    parser.add_argument("--slices", type=int, nargs='+', default=[1, 3, 6, 12], help="List of slice counts")
    parser.add_argument("--niter", type=int, default=20, help="Number of iterations")
    parser.add_argument("--save", type=int, default=10000, help="Number of iterations to save the result")
    parser.add_argument(
        "--compile", 
        nargs="?",         # Consumes 0 or 1 arguments
        const="default",   # Value if --compile is present but empty
        default=False,     # Value if --compile is entirely absent
        help="Enable compilation. If passed without a string, defaults to 'default'. Can also pass a mode like 'max-autotune' or 'reduce-overhead'."
    )
    
    args = parser.parse_args()
    
    # Calculate the total size of the benchmark grid
    total_runs = len(args.round_idx) * len(args.batches) * len(args.pmodes) * len(args.slices)
    
    # ---------------------------------------------------------
    # Clean Configuration Printout
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("### Benchmark Configuration ###")
    print("="*50)
    
    # vars(args) turns the namespace into a dictionary we can loop through cleanly
    for key, value in vars(args).items():
        # The :<12 aligns the text so your log looks like a neat table
        print(f" {key:<12}: {value}")
        
    print("-" * 50)
    print(f" Total Benchmark Runs Scheduled: {total_runs}")
    print("="*50 + "\n")
    # ---------------------------------------------------------
    
    for round_idx, batch, pmode, zslice in itertools.product(args.round_idx, args.batches, args.pmodes, args.slices):
        
        try:
            # Setup LoggingManager for individual runs
            LoggingManager(
                log_file='ptyrad_log.txt',
                log_dir='auto',
                prefix_time='datetime',
                prefix_jobid=0,
                append_to_file=True,
                show_timestamp=True,
                verbosity='INFO'
            )

            print_system_info()
            
            init_params = InitParams(
                probe_kv=80,
                probe_conv_angle=24.9,
                meas_Npix=128,
                pos_N_scan_slow=128,
                pos_N_scan_fast=128,
                pos_scan_step_size=0.4290,
                meas_calibration={'mode': 'dx', 'value': 0.1494},
                probe_pmode_max=pmode,
                obj_Nlayer=zslice,
                obj_slice_thickness=round(12/zslice, 2),
                meas_flipT=[1,0,0],
                pos_scan_affine=[1,0,-3,0],
                meas_params={'path': 'data/tBL_WSe2/Panel_g-h_Themis/scan_x128_y128.raw'}
            )

            params = PtyRADParams(init_params=init_params).model_dump() # dump it back to normal python dict for easy modification
            
            device = set_gpu_device(args.gpuid)
                            
            # Run ptyrad
            print(f"### Running (round_idx, batch, pmode, slice) = {(round_idx, batch, pmode, zslice)} ###")
            
            output_dir = f"output/tBL_WSe2/{args.date}_{args.label}_{args.device}_table_r{str(round_idx)}/"
            params['recon_params']['output_dir'] = output_dir
            params['recon_params']['NITER'] = args.niter
            params['recon_params']['SAVE_ITERS'] = args.save
            params['recon_params']['BATCH_SIZE']['size'] = batch
            params['recon_params']['recon_dir_affixes'] = 'minimal'
            params['recon_params']['prefix_time'] = False
            
            if args.compile:
                compiler_configs = {'enable': True, 'mode': args.compile}
            else:
                compiler_configs = {'enable': False}

            params['recon_params']['compiler_configs'] = compiler_configs
            
            ptycho_solver = PtyRADSolver(params, device=device)
            ptycho_solver.run()
            
        except Exception as e:
            print(f"An error occurred for (round, batch, pmode, slice) = {(round_idx, batch, pmode, zslice)}: {e}")

        finally:
            del ptycho_solver  # Clear model
            torch.cuda.empty_cache()
            gc.collect()

