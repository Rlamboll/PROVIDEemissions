import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

version = "v6"
run_scenarios = "chosen_files"
fairdir = '../output/{}/{}/fair_{}/'.format(run_scenarios, version, "temperatures")
summaryname = "summary.csv"
scenariofiles = [
        x for x in os.listdir(fairdir)
        if x.endswith('.csv') and x != summaryname
]

results = []
quantiles = [0.1, 0.33, 0.5, 0.66, 0.9]
for scen in scenariofiles:
    if scen in ["Abrupt_stop", "Current policies"]:
        continue
    tmp = pd.read_csv(fairdir + scen, index_col="year")
    for q in quantiles:
        quant_res = tmp.quantile(q, axis=1)
        quant_res = pd.DataFrame(quant_res).T
        quant_res["quantile"] = q
        quant_res["scenario"] = scen[5:-4]
        results.append(quant_res)

results = pd.concat(results)
scen1 = results.scenario.iloc[0]
results = results.set_index(["scenario", "quantile"])

# Construct pathways that freeze in temp at earlier points.
# Neg is frozen at peak, other scenarios frozen at 2100
freeze_scen = ["CurPol", "ModAct", "Neg"]
for scen in freeze_scen:
    new_result = results.loc[scen, :]
    freeze_year = min(new_result.idxmax(axis=1)[0.50], 2100)
    new_result.loc[:, freeze_year:] = np.repeat(new_result.loc[
                                                :, freeze_year:freeze_year
    ].values, new_result.loc[:, freeze_year:].shape[1], axis=1)
    new_result.index = [(scen + "_SAP", i) for i in new_result.index]
    results = results.append(new_result)

# Construct pathways that freeze at certain points.
# Base 1.5 C pathway on the first scenario
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

to_plot = results.loc[results.index.get_level_values("quantile")==0.5, :]
labels = to_plot.index.get_level_values("scenario")
plt.plot(results.columns, to_plot.T)
plt.legend(labels, bbox_to_anchor=(1.05, 1))
plt.xlabel("Year")
plt.ylabel("Warming ($^o$C)")
plt.savefig(fairdir + "plot0.5quant.png", bbox_inches="tight")

