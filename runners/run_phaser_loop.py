# Python script to run phaser
# Updated by Chia-Hao Lee on 2026.03.11

import argparse
import gc
import os
import itertools
import datetime

import jax

from phaser.plan import ReconsPlan
from phaser.execute import execute_plan

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
            description="Run phaser benchmark loop", 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    
    # Generate today's date in YYYYMMDD format (e.g., "20260310")
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    
    # Benchmark args
    parser.add_argument("--gpuid", type=int, default=0, help="GPU ID to use, default is 0")
    parser.add_argument("--device", type=str, required=True, help="Device string for output folder (e.g., RTX3080 or A100_20GB)")
    parser.add_argument("--date", type=str, default=today_str, help="Date string for output folder (e.g., 20260310)")
    parser.add_argument("--label", type=str, default='phaser', help="Arbitrary label string for identification")
    parser.add_argument("--round_idx", type=int, nargs='+', default=[1, 2, 3, 4, 5], help="List of round indices")
    parser.add_argument("--batches", type=int, nargs='+', default=[1024, 512, 256, 128, 64, 32, 16, 8, 4], help="List of batch sizes")
    parser.add_argument("--pmodes", type=int, nargs='+', default=[1, 3, 6, 12], help="List of probe modes")
    parser.add_argument("--slices", type=int, nargs='+', default=[1, 3, 6, 12], help="List of slice counts")
    parser.add_argument("--niter", type=int, default=20, help="Number of iterations")
    parser.add_argument("--save", type=int, default=10000, help="Number of iterations to save the result")
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
            
            devices = jax.devices('gpu')
            print(f"Available devices: {list(devices)}")
            desired_device = devices[args.gpuid] 
            jax.config.update("jax_default_device", desired_device)
            print(f"Default device: {jax.config.jax_default_device} ({desired_device.device_kind})")    
            
            # Run phaser
            print(f"### Running (round_idx, batch, pmode, slice) = {(round_idx, batch, pmode, zslice)} ###")
            
            output_dir = f"output/tBL_WSe2/{args.date}_{args.label}_{args.device}_table_r{str(round_idx)}/b{batch}_p{pmode}_{zslice}slice/"
            os.makedirs(output_dir, exist_ok=True)
            
            params = {
                "name": output_dir,
                "backend": "jax",
                # raw data source
                "raw_data": {
                    "type": "empad",
                    "path": "data/tBL_WSe2/Panel_g-h_Themis/tBL_WSe2.json",
                    "det_flips": [True, False, False],
                },
                # initialization
                "slices": {
                    "n": zslice,
                    "total_thickness": 12,
                },
                "engines": [
                    {
                        "type": "gradient",
                        "sim_shape": [128, 128],
                        "probe_modes": pmode,
                        "niter": args.niter,
                        "grouping": batch,
                        "bwlim_frac": 1.0,
                        "noise_model": {
                            "type": "anscombe",
                            "eps": 0.1,
                        },
                        "solvers": {
                            "object": {
                                "type": "adam",
                                "learning_rate": 5.0e-3,
                                "nesterov": True,
                            },
                            "probe": {
                                "type": "adam",
                                "learning_rate": 0.1,
                                "nesterov": True,
                            },
                            "positions": {
                                "type": "sgd",
                                "learning_rate": 0.5,
                                "momentum": 0.90,
                                "nesterov": True,
                            },
                        },
                        # Regularizers (soft constraint) will cause extra compute in loss. Disable it since PtyRAD is only using loss_single.
                        "regularizers": [
                            # {"type": "obj_l2", "cost": 0.4},
                            # {"type": "obj_tikh", "cost": 0.2},
                            # {"type": "layers_tikh", "cost": 5.0e2},
                        ],
                        "group_constraints": [],
                        "iter_constraints": [
                            {"type": "limit_probe_support", "max_angle": 25.0},
                            {"type": "clamp_object_amplitude", "amplitude": 1.0},
                            {"type": "layers", "sigma": 100.0, "weight": 0.8}
                        ],
                        "save": {"every": args.save},
                        "save_images": {"every": args.save},
                        "save_options": {
                            "images": [
                                "probe",
                                "probe_recip",
                                "object_phase_sum",
                                "object_mag_sum",
                                "object_phase_stack",
                                "object_mag_stack",
                            ]
                        },
                    }
                ],
            }
            plan = ReconsPlan.from_data(params)
            execute_plan(plan)

        except Exception as e:
            print(f"An error occurred for (round, batch, pmode, slice) = {(round_idx, batch, pmode, zslice)}: {e}")
            
        finally:
            jax.clear_caches()
            gc.collect()