# Python script to run phaser
# Updated by Chia-Hao Lee on 2026.03.08

# import argparse
import gc
import os

import jax

from phaser.plan import ReconsPlan
from phaser.execute import execute_plan

if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(
    #     description="Run phaser", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )
    # parser.add_argument("--params_path", type=str, required=True)
    # args = parser.parse_args()
    
    for round_idx in range(1, 6):
        for batch in [1024, 512, 256, 128, 64, 32, 16, 8, 4]:
            for pmode in [1,3,6,12]:
                for slice in [1,3,6,12]:
                    try:
                        
                        # Run phaser
                        print(f"### Running (round_idx, batch, pmode, slice) = {(round_idx, batch, pmode, slice)} ###")
                        
                        output_dir = f"output/tBL_WSe2/20260308_phaser_20GB_table_r{str(round_idx)}/b{batch}_p{pmode}_{slice}slice/"
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
                                "n": slice,
                                "total_thickness": 12,
                            },
                            "engines": [
                                {
                                    "type": "gradient",
                                    "sim_shape": [128, 128],
                                    "probe_modes": pmode,
                                    "niter": 20,
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
                                    "save": {"every": 20},
                                    "save_images": {"every": 20},
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
                        print(f"An error occurred for (round, batch, pmode, slice) = {(round_idx, batch, pmode, slice)}: {e}")
                        
                    finally:
                        jax.clear_caches()
                        gc.collect()
