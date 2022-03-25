import json
from multiprocessing import Pool, cpu_count, freeze_support

from climateforcing.utils import mkdir_p
import fair
import numpy as np
import pandas as pd
from tqdm import tqdm

with open('../RunFaIRAR6/data_input/fair-1.6.2-ar6/fair-1.6.2-wg3-params.json') as f:
    config_list = json.load(f)

emissions_in = {}
results_out = {}
WORKERS = cpu_count() - 1
version = "v7_5"
outdir = '../output/{}/fair_{}/'

"""
scenarios = ["ssp245_constant-2020-ch4", "ch4_30", "ch4_40", "ch4_50", "coal-phase-out"]

for scenario in scenarios:
    emissions_in[scenario] = np.loadtxt('../RunFaIRAR6/data_output/fair_emissions_files/{}.csv'.format(scenario), delimiter=',')
"""

def run_fair(args):
    thisC, thisF, thisT, _, thisOHU, _, thisAF = fair.forward.fair_scm(**args)
    return (thisC[:, 0], thisC[:, 1], thisT, thisF[:, 1], np.sum(thisF, axis=1))


def fair_process(emissions):
    updated_config = []
    for i, cfg in enumerate(config_list):
        updated_config.append({})
        for key, value in cfg.items():
            if isinstance(value, list):
                updated_config[i][key] = np.asarray(value)
            else:
                updated_config[i][key] = value
        updated_config[i]['emissions'] = emissions
        updated_config[i]['diagnostics'] = 'AR6'
        updated_config[i]["efficacy"] = np.ones(45)
        updated_config[i]["gir_carbon_cycle"] = True
        updated_config[i]["temperature_function"] = "Geoffroy"
        updated_config[i]["aerosol_forcing"] = "aerocom+ghan2"
        updated_config[i]["fixPre1850RCP"] = False
        #    updated_config[i]["scale"][43] = 0.6
        updated_config[i]["F_solar"][270:] = 0

    # multiprocessing is not working for me on Windows
    if __name__ == '__main__':
        with Pool(WORKERS) as pool:
            result = list(
                tqdm(pool.imap(run_fair, updated_config), total=len(updated_config),
                     position=0, leave=True))

        result_t = np.array(result).transpose(1, 2, 0)
        c_co2, c_ch4, t, f_ch4, f_tot = result_t
        temp_rebase = t - t[100:151, :].mean(axis=0)
    else:
        raise RuntimeError

    return c_co2, c_ch4, temp_rebase, f_ch4, f_tot

def main():
    freeze_support()
    for scenario in tqdm(scenarios, position=0, leave=True):
        results_out[scenario] = {}
        (
            results_out[scenario]['co2_concentrations'],
            results_out[scenario]['ch4_concentrations'],
            results_out[scenario]['temperatures'],
            results_out[scenario]['ch4_effective_radiative_forcing'],
            results_out[scenario]['effective_radiative_forcing']
        ) = fair_process(emissions_in[scenario])

    for scenario in scenarios:
        for var in [
            'co2_concentrations', 'ch4_concentrations', 'temperatures',
            'ch4_effective_radiative_forcing', 'effective_radiative_forcing'
        ]:
            mkdir_p(outdir.format(version, var))
            df_out = pd.DataFrame(results_out[scenario][var][260:351, :])
            df_out['year'] = np.arange(2010, 2101)
            df_out.set_index('year', inplace=True)
            df_out.to_csv(outdir.format(version, var) + '{}.csv'.format(scenario), float_format="%6.4f")

if __name__ == "__main__":
    main()