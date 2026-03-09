# Python script to run PtyRAD
# Updated by Chia-Hao Lee on 2026.03.08

# import argparse
import gc
import os
from importlib.metadata import version  # Temporary fix for version check
from pathlib import Path

import numpy as np
import torch
from quantem.core import config
from quantem.core.datastructures import Dataset4dstem
from quantem.diffractive_imaging.benchmark_utils import load_raw

print(f"quantEM version: {version("quantem")}")

from quantem.diffractive_imaging import (
    DetectorPixelated,
    ObjectPixelated,
    ProbePixelated,
    Ptychography,
    PtychographyDatasetRaster,
)


if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(
    #     description="Run quantem", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )
    # parser.add_argument("--params_path", type=str, required=True)
    # parser.add_argument("--skip_validate", action="store_true", help="Skip parameter validation and default filling. Use only if your params file is complete and consistent.")
    # parser.add_argument("--gpuid", type=str, required=False, default="0", help="GPU ID to use ('acc', 'cpu', or an integer)")
    # parser.add_argument("--jobid", type=int, required=False, default=0, help="Unique identifier for hypertune mode with multiple GPU workers")
    # args = parser.parse_args()
    
    for round_idx in range(1, 6):
        for batch in [1024, 512, 256, 128, 64, 32, 16, 8, 4]:
            for pmode in [1,3,6,12]:
                for slice in [1,3,6,12]:
                    try:
                        
                        GPU_ID = 0
                        config.set_device(GPU_ID)
                        print(f"device set to {config.get("device")}")
                        
                        # Run quantem
                        print(f"### Running (round_idx, batch, pmode, slice) = {(round_idx, batch, pmode, slice)} ###")
                        
                        output_dir = f"output/tBL_WSe2/20260308_quantem_20GB_table_r{str(round_idx)}/b{batch}_p{pmode}_{slice}slice/"
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
                        NUM_SLICES = slice 
                        SLICE_THICKNESS = round(12/slice, 2) # Ang
                        
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
                            num_iters=20,
                            reset=True,
                            autograd=True, 
                            device=config.get("device"),
                            constraints=constraints, 
                            optimizer_params=opt_params,
                            scheduler_params=scheduler_params,
                            batch_size=BATCH_SIZE,
                        )
                        
                        # Save zip
                        ptycho_pix.save(f"{output_dir}/ptycho_pix.zip", mode='o')
                        
                    except Exception as e:
                        print(f"An error occurred for (round, batch, pmode, slice) = {(round_idx, batch, pmode, slice)}: {e}")

                    finally:
                        del data, dset, pdset, ptycho_pix # Clear data and model
                        torch.cuda.empty_cache()
                        gc.collect()

