import numpy as np
import scripts.generate_systematic_paths
version = "v6"
outdir = f"../output/{version}/"

net_zeros = [2040, 2050, 2060, 2070, 2080, 2090, 2100, 2150, 2200]
overshoots = [0, -2500, -5000, -7500, -10000]
mod_2030s = np.arange(0.12, 1.32, 0.2)
methane_levels = [0.1, 0.25, 0.5, 0.75, 0.9]

cases = []
for mod_2030 in mod_2030s:
    for net_zero in net_zeros:
        for overshoot in overshoots:
            for methane in methane_levels:
                cases.append((mod_2030, net_zero, overshoot, methane))

scripts.generate_systematic_paths.make_paths(
    cases=cases,
    outdir=outdir,
    version=version,
    check_min_dif=True, make_scen_files=False
)