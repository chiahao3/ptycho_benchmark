# Python script to run quantem
# Updated by Chia-Hao Lee on 2026.03.11

import argparse
import datetime
import gc
import os
from importlib.metadata import version  # Temporary fix for version check
import itertools
from pathlib import Path

import numpy as np
import torch
from quantem.core import config
from quantem.core.datastructures import Dataset4dstem
from quantem.diffractive_imaging.benchmark_utils import load_raw
from quantem.diffractive_imaging import (
    DetectorPixelated,
    ObjectPixelated,
    ProbePixelated,
    Ptychography,
    PtychographyDatasetRaster,
)
print(f"quantEM version: {version("quantem")}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
            description="Run quantem benchmark loop", 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    
    # Generate today's date in YYYYMMDD format (e.g., "20260310")
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    
    # Benchmark args
    parser.add_argument("--gpuid", type=int, default=0, help="GPU ID to use, default is 0")
    parser.add_argument("--device", type=str, required=True, help="Device string for output folder (e.g., RTX3080 or A100_20GB)")
    parser.add_argument("--date", type=str, default=today_str, help="Date string for output folder (e.g., 20260310)")
    parser.add_argument("--label", type=str, default='quantem', help="Arbitrary label string for identification")
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
            
            config.set_device(args.gpuid)
            print(f"device set to {config.get("device")}")
            
            # Run quantem
            print(f"### Running (round_idx, batch, pmode, slice) = {(round_idx, batch, pmode, zslice)} ###")
            
            output_dir = f"output/tBL_WSe2/{args.date}_{args.label}_{args.device}_table_r{str(round_idx)}/b{batch}_p{pmode}_{zslice}slice/"
            os.makedirs(output_dir, exist_ok=True)
            
            # Load 4D dataset
            fdata = Path("data/tBL_WSe2/Panel_g-h_Themis/scan_x128_y128.raw")
            data = load_raw(fdata, shape=(16384, 128, 128))
            data = np.flip(data, axis=1) # k-space flip up-down
            data = data.reshape((128,128,128,128)).copy()
            # data = data[64:,64:]
            data = data.clip(0)
            data = data / 151 # The convergence speed is dependent on probe / data normalization if using a fixed learning rate
            
            dset = Dataset4dstem.from_array(
                array=data,
                sampling=[0.429, 0.429, 0.0523, 0.0523],
                origin=[0,0,0,0],
                units=['A', 'A', 'A^-1', 'A^-1'],
            )
            
            # Set params
            PROBE_ENERGY = 80e3
            PROBE_SEMIANGLE = 24.9 # mrad 
            PROBE_DEFOCUS = 0 # Ang 

            BATCH_SIZE = batch
            NUM_PROBES = pmode
            NUM_SLICES = zslice 
            SLICE_THICKNESS = round(12/zslice, 2) # Ang
            
            # Initialize dataset
            pdset = PtychographyDatasetRaster.from_dataset4dstem(
                dset,
                learn_scan_positions=True,
                learn_descan=False,
                verbose=False # # This will disable tqdm progress bar
            )

            pdset.preprocess(
                com_fit_function="constant",
                plot_rotation=False,
                plot_com=False,
                probe_energy=PROBE_ENERGY,
                force_com_transpose=False,
                vectorized=True,
            )    
            
            # Initialize iterative ptycho and component models
            obj_model_pix = ObjectPixelated.from_uniform(
                num_slices=NUM_SLICES, 
                slice_thicknesses=SLICE_THICKNESS,
                obj_type='complex',
                device='cuda'
            )

            probe_params = {
                "energy" : PROBE_ENERGY,
                "defocus": PROBE_DEFOCUS,
                # "aberration_coefs": {'C10': 0},
                "semiangle_cutoff" : PROBE_SEMIANGLE, 
            }

            probe_model_pix = ProbePixelated.from_params(
                num_probes=NUM_PROBES,
                probe_params=probe_params,
                device='cuda'
            )

            detector_model = DetectorPixelated() 

            ptycho_pix = Ptychography.from_models(
                dset=pdset,
                obj_model=obj_model_pix,
                probe_model=probe_model_pix,
                detector_model=detector_model,
                device='cuda',
                verbose=False, # This will disable tqdm progress bar
            )

            ptycho_pix.preprocess( 
                obj_padding_px=(123,123), # Pad it such that quantem is roughly using the same object size (614,614) with PtyRAD
                batch_size=BATCH_SIZE,
            )
            
            # Set optimizer, scheduler, and constraint params
            # NOTE: Not sure if these are good settings, need to ask Arthur / Colin O
            opt_params = {
                    "object": {
                        "type": "adamw", 
                        "lr": 5e-3, 
                    },
                    "probe": {
                        "type": "adamw", 
                        "lr": 5e-3, 
                    },
                    "dataset": { ### for optimizing over descan shifts and probe positions
                        "type": "adamw",
                        "lr": 1e-4,
                    }
            }

            scheduler_params = {
                "object": { ## scheduler kwargs are passed to the scheduler (of type type)
                    "type": "plateau", ## i like plateau for many cases
                },
                "probe": {
                    "type": "plateau",
                },
                "dataset": { 
                    "type": "plateau",  ## exp is also frequently used 
                }
            }

            constraints = {
                "object": {
                    # "tv_weight_z": 10, # NOTE: Really not sure about the proper weight for this. Disable since soft constraint will cause extra compute and PtyRAD is only doing loss_single
                },
                "dataset": {
                    "center_scan_positions": True,
                }
            }

            # Run reconstruction
            ptycho_pix.reconstruct(
                num_iters=args.niter,
                reset=True,
                autograd=True, 
                device=config.get("device"),
                constraints=constraints, 
                optimizer_params=opt_params,
                scheduler_params=scheduler_params,
                batch_size=BATCH_SIZE,
            )
            
            # Save final zip
            if args.save <= args.niter:
                ptycho_pix.save(f"{output_dir}/ptycho_pix.zip", mode='o')
            
        except Exception as e:
            print(f"An error occurred for (round, batch, pmode, slice) = {(round_idx, batch, pmode, zslice)}: {e}")

        finally:
            del data, dset, pdset, ptycho_pix # Clear data and model
            torch.cuda.empty_cache()
            gc.collect()