import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pyam
import re
import silicone.utils

version = "v8"
run_scenarios = "chosen_files"
outdir = f"../output/{run_scenarios}/{version}/"
fairdir = '../output/{}/{}/fair_{}/'.format(run_scenarios, version, "temperatures")
plotdir = outdir + "plots/"
if not os.path.exists(plotdir):
    os.makedirs(plotdir)
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

scens_to_restrict = [
    ("CurPol nz 2120 decline -23500", 1.51),
    ("ModAct nz 2120 decline -23000", 1),
    ("Neg nz 2100 decline -17000", 0),
    ("ModAct nz 2120 decline -16500", 1.5),
]

scen_to_remove_later = [i[0] for i in scens_to_restrict]

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
freeze_dates = {}
for scen in freeze_scen:
    new_result = results.loc[scen, :]
    freeze_year = min(new_result.idxmax(axis=1)[0.50], 2100)
    new_result.loc[:, freeze_year:] = np.repeat(new_result.loc[
                                                :, freeze_year:freeze_year
    ].values, new_result.loc[:, freeze_year:].shape[1], axis=1)
    new_result.index = [(scen + "_SAP", i) for i in new_result.index]
    results = results.append(new_result)
    freeze_dates[scen] = freeze_year

# Construct pathways that freeze at certain temperatures after 2100.
# Base 1.5 C pathway on the chosen scenario in scen1
assert max(results.loc[(scen1, 0.5)]) > 1.5
first_ind_15 = results.loc[(scen1, 0.5)] < 1.5
first_ind_15_ends = np.cumprod(first_ind_15)
new_res = results.loc[scen1, [bool(x) for x in first_ind_15_ends.values]]
new_cols = {}
for col in [c for c in results.columns if c not in new_res]:
    new_cols[col] = new_res.iloc[:, -1]
new_res = pd.concat([new_res, pd.DataFrame(new_cols)], axis=1)
new_res = new_res.reset_index(drop=False)
new_res["scenario"] = "Ref_1p5"
new_res = new_res.set_index(["scenario", "quantile"])
results = results.append(new_res)

post_2100 = [c for c in results.columns if c > 2100]
for (scen, temp) in scens_to_restrict:
    assert results.loc[(scen, 0.5), post_2100].values.min() < temp, \
        f"Scenario {scen} never goes below {temp}"
    first_ind_15 = (results.loc[(scen, 0.5), :] > temp) | (results.columns < 2100)
    first_ind_15_ends = np.cumprod(first_ind_15)
    new_res = results.loc[scen, [bool(x) for x in first_ind_15_ends.values]]
    for col in [c for c in results.columns if c not in new_res]:
        new_res[col] = new_res.iloc[:, -1]
    new_res = new_res.reset_index(drop=False)
    new_res["scenario"] = scen.split(" ")[0] + f"_OS_{temp}C"
    new_res = new_res.set_index(["scenario", "quantile"])
    results = results.append(new_res)

results = results.loc[
    [s not in scen_to_remove_later for s in results.index.get_level_values("scenario")]
]
results.to_csv(fairdir + summaryname)

emissions = pyam.IamDataFrame(outdir + "all_emissions.csv")
emissions = emissions.filter(scenario=scen_to_remove_later, keep=False)
co2 = "Emissions|CO2"
co2s = [co2 + "|AFOLU", co2 + "|Energy and Industrial Processes"]
co2tot = silicone.utils._construct_consistent_values(
    co2, co2s, emissions
).timeseries().reset_index()
ghgs = co2s + ["*CH4", "*F-Gases*", "*SF6", "*HFC*", "*PFC*", "*N2O*"]
ghgtot = silicone.utils._construct_consistent_values(
    "Kyoto GHG total (AR6GWP100)", ghgs, silicone.utils.convert_units_to_MtCO2_equiv(
        emissions.filter(variable=ghgs).rename(
            {"unit":{'kt HFC43-10/yr': 'kt HFC4310/yr'}}
        ), metric_name="AR5GWP100")
).timeseries().reset_index()

# Label the scenarios with the data to plot them nicely
def file_scenarios(medtemps):
    atomic_scen = [i.split(" ")[0].split("_")[0] for i in medtemps["scenario"]]
    isovershoot = [any([re.search("OS", i)]) for i in medtemps["scenario"]]
    isnz = [any([re.search("NZ", i)]) for i in medtemps["scenario"]]
    medtemps["atomic_scen"] = atomic_scen
    medtemps["linestyle"] = "-"
    medtemps.loc[isovershoot, "linestyle"] = "--"
    medtemps.loc[medtemps["scenario"] == "ModAct_OS_1C", "linestyle"] = ":"
    medtemps.loc[isnz, "linestyle"] = "-."
    return (atomic_scen, isovershoot, isnz, medtemps)
atomic_scen, isovershoot, isnz, results = file_scenarios(results.reset_index())
_, _, _, ghgtot = file_scenarios(ghgtot)
_, _, _, co2tot = file_scenarios(co2tot)

# Design a color scheme
cmap = plt.get_cmap('tab10')
colors = cmap(np.linspace(0, 1, len(set(atomic_scen))))
scenset = list(set(atomic_scen))
cdict = {scenset[i]: colors[i] for i in range(len(scenset))}

results = results.sort_values(2100, ascending=False)

# Process files for both before and adter 2100
for include in [True, False]:
    if include:
        years = np.arange(2010, 2300)
        emissions_years = [2015] + list(np.arange(2020, 2301, 10))
        to_plot = results.loc[
          (results["quantile"]==0.5) & [i not in scens_for_2100 for i in results["scenario"]],
          :
        ]
        savestring = "post-2100"
        co2date = co2tot.loc[[i not in scens_for_2100 for i in co2tot["scenario"]], :]
        ghgtotdate = ghgtot.loc[[i not in scens_for_2100 for i in ghgtot["scenario"]], :]
    else:
        years = np.arange(2010, 2100)
        emissions_years = [2015] + list(np.arange(2020, 2101, 10))
        to_plot = results.loc[
          (results["quantile"] == 0.5) & (
              [i in (scens_for_2100 + scenes_for_both) for i in results["scenario"]]
          ),
          ["scenario", "linestyle", "atomic_scen"] + list(years)
        ]
        co2date = co2tot.loc[
            [i in (scens_for_2100 + scenes_for_both) for i in co2tot["scenario"]], :
        ]
        ghgtotdate = ghgtot.loc[
            [i in (scens_for_2100 + scenes_for_both) for i in ghgtot["scenario"]], :
        ]
        savestring = "until-2100"
    labels = to_plot["scenario"]
    plt.clf()
    for (j, scen) in to_plot.iterrows():
        plt.plot(
            years, scen.loc[years], c=cdict[scen["atomic_scen"]],
            linestyle=scen["linestyle"]
        )
    plt.legend(to_plot["scenario"], bbox_to_anchor=(1.02, 1))
    plt.xlabel("Year")
    plt.ylabel("Temperature ($^o$C)")
    plt.savefig(plotdir + savestring + "plot0.5quant.png", bbox_inches="tight")
    to_plot.to_csv(fairdir + savestring + summaryname)
    # Then plot the emissions totals
    plt.clf()
    for (j, scen) in co2date.iterrows():
        plt.plot(
            emissions_years, scen.loc[emissions_years], c=cdict[scen["atomic_scen"]],
            linestyle=scen["linestyle"]
        )
    plt.legend(co2date["scenario"], bbox_to_anchor=(1.02, 1))
    plt.xlabel("Year")
    plt.ylabel("CO$_2$ emissions (Gt CO$_2$/yr)")
    plt.savefig(plotdir + savestring + "plotco2emissions.png", bbox_inches="tight")
    plt.clf()
    for (j, scen) in ghgtotdate.iterrows():
        plt.plot(
            emissions_years, scen.loc[emissions_years], c=cdict[scen["atomic_scen"]],
            linestyle=scen["linestyle"]
        )
    plt.legend(co2date["scenario"], bbox_to_anchor=(1.02, 1))
    plt.xlabel("Year")
    plt.ylabel("Kyoto GHG emissions (Gt CO$_2$-eq/yr)")
    plt.savefig(plotdir + savestring + "plotghgemissions.png", bbox_inches="tight")



