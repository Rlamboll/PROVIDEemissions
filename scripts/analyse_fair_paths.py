import numpy as np
import pandas as pd
import os

version = "v5"
run_scenarios = "selected"
fairdir = '../output/{}/{}/fair_{}/'.format(version, run_scenarios, "temperatures")
summaryname = "summary.csv"
scenariofiles = [
        x for x in os.listdir(fairdir)
        if x.endswith('.csv') and x != summaryname
]

results = []
quantiles = [0.1, 0.33, 0.5, 0.66, 0.9]
for scen in scenariofiles:
    tmp = pd.read_csv(fairdir + scen, index_col="year")
    for q in quantiles:
        quant_res = tmp.quantile(q, axis=1)
        quant_res = pd.DataFrame(quant_res).T
        quant_res["quantile"] = q
        quant_res["scenario"] = scen[:-4]
        results.append(quant_res)

results = pd.concat(results)
scen1 = results.scenario.iloc[0]
results = results.set_index(["scenario", "quantile"])

# Construct a pathway that stops at 1.5 C.
# Base this on the first scenario
assert max(results.loc[(scen1, 0.5)]) > 1.5
first_ind_15 = results.loc[(scen1, 0.5)] < 1.5
first_ind_15_ends = np.cumprod(first_ind_15)
new_res = results.loc[scen1, [bool(x) for x in first_ind_15_ends.values]]
for col in [c for c in results.columns if c not in new_res]:
    new_res[col] = new_res.iloc[:, -1]
new_res = new_res.reset_index(drop=False)
new_res["scenario"] = "Costructed 1.5 C"
new_res = new_res.set_index(["scenario", "quantile"])
results = results.append(new_res)
results.to_csv(fairdir + summaryname)


