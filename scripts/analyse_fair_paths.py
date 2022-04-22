import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

version = "v7"
run_scenarios = "chosen_files"
fairdir = '../output/{}/{}/fair_{}/'.format(run_scenarios, version, "temperatures")
summaryname = "summary.csv"
scenariofiles = [
        x for x in os.listdir(fairdir)
        if x.endswith('.csv') and x != summaryname
]

# These scenarios are simply a subset of other scenarios
unused_scenarios = ["until-2100summary.csv", "post-2100summary.csv"]

scens_for_2100 = [
    "GS", "Neg", "ModAct", "CurPol", "LD", "SP", "Ren",
]
scenes_for_both = ["ssp119", "ssp534-over", "Ref_1p5"]

scen_to_remove_later = [
    "ModAct nz 2120 decline -22000",
    "Neg nz 2100 decline -17000",
    "ModAct nz 2120 decline -17000",
    "ModAct nz 2120 decline -23000",
    "Neg nz 2100 decline -20000",
]
results = []
quantiles = [0.1, 0.2, 0.3, 0.33, 0.4, 0.5, 0.6, 0.66, 0.7, 0.8, 0.9]
for scen in scenariofiles:
    if scen in unused_scenarios:
        continue
    tmp = pd.read_csv(fairdir + scen, index_col="year")
    for q in quantiles:
        quant_res = tmp.quantile(q, axis=1)
        quant_res = pd.DataFrame(quant_res).T
        quant_res["quantile"] = q
        quant_res["scenario"] = scen[5:-4]
        results.append(quant_res)

results = pd.concat(results)
scen1 = "LD"
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

# Construct pathways that freeze at certain temperatures after 2100.
# Base 1.5 C pathway on the chosen scenario in scen1
assert max(results.loc[(scen1, 0.5)]) > 1.5
first_ind_15 = results.loc[(scen1, 0.5)] < 1.5
first_ind_15_ends = np.cumprod(first_ind_15)
new_res = results.loc[scen1, [bool(x) for x in first_ind_15_ends.values]]
for col in [c for c in results.columns if c not in new_res]:
    new_res[col] = new_res.iloc[:, -1]
new_res = new_res.reset_index(drop=False)
new_res["scenario"] = "Ref_1p5"
new_res = new_res.set_index(["scenario", "quantile"])
results = results.append(new_res)

post_2100 = [c for c in results.columns if c > 2100]
for (scen, temp) in [
    ("ModAct nz 2120 decline -22000", 1),
    ("Neg nz 2100 decline -17000", 0),
    ("ModAct nz 2120 decline -17000", 1.5),
]:
    assert min(results.loc[(scen, 0.5), post_2100]) < temp, \
        f"Scenario {scen} never goes below {temp}"
    first_ind_15 = (results.loc[(scen, 0.5), :] > temp) | (results.columns < 2100)
    first_ind_15_ends = np.cumprod(first_ind_15)
    new_res = results.loc[scen, [bool(x) for x in first_ind_15_ends.values]]
    for col in [c for c in results.columns if c not in new_res]:
        new_res[col] = new_res.iloc[:, -1]
    new_res = new_res.reset_index(drop=False)
    new_res["scenario"] = scen + f"_to_{temp}"
    new_res = new_res.set_index(["scenario", "quantile"])
    results = results.append(new_res)

results = results.loc[
    [s not in scen_to_remove_later for s in results.index.get_level_values("scenario")]
]
results.to_csv(fairdir + summaryname)
for include in [True, False]:
    if include:
        to_plot = results.loc[
                  (results.index.get_level_values("quantile")==0.5) & (
                    [i not in scens_for_2100 for i in results.index.get_level_values("scenario")]
                  ),
                  :
        ]
        savestring = "post-2100"
    else:
        to_plot = results.loc[
                  (results.index.get_level_values("quantile") == 0.5) & (
                      [i in (scens_for_2100 + scenes_for_both) for i in
                       results.index.get_level_values("scenario")]),
                  [c for c in results.columns if c <= 2100]
                  ]
        savestring = "until-2100"
    labels = to_plot.index.get_level_values("scenario")
    plt.clf()
    plt.plot(to_plot.columns, to_plot.T)
    plt.legend(labels, bbox_to_anchor=(1.05, 1))
    plt.xlabel("Year")
    plt.ylabel("Warming ($^o$C)")
    plt.savefig(fairdir + savestring + "plot0.5quant.png", bbox_inches="tight")
    to_plot.to_csv(fairdir + savestring + summaryname)

