import pyam
import pandas as pd
import numpy as np

version = "v8"
run_scenarios = "chosen_files"
outdir = f"../output/{run_scenarios}/{version}/"
fairdir = '../output/{}/{}/fair_{}/'.format(run_scenarios, version, "temperatures")

temps = pd.read_csv(fairdir + "until-2100_inc_redsummary.csv").set_index("scenario").iloc[:, 3:]

emissions = pyam.IamDataFrame(outdir + "all_emissions.csv")
co2 = "Emissions|CO2"
interesting = {"Max temp": temps.max(axis=1)}
stylised = pd.Series(
    index=temps.index, data=["Plausible"] * 10
)
stylised[["Ref_1p5", "ssp534-over"]] = "Stylised"
interesting["Stylised"] = stylised
interesting["Warming 2050"] = temps["2050"]
interesting["Warming 2100"] = temps["2100"]
co2ems = emissions.aggregate(co2).filter(variable=co2)
for year in [x for x in range(2020, 2050) if x not in co2ems.year]:
    co2ems = co2ems.interpolate(year)
co2cumsums = np.cumsum(co2ems.timeseries(), axis=1) - co2ems.timeseries()[2015]
interesting["Total emissions until 2050"] = co2cumsums[2050]
interesting["Total emissions until 2100"] = co2cumsums[2100]
exceeds = {}
for exceedence in [1.5, 2.0, 3.0, 4.0]:
    for scenname in temps.index:
        allruns = pd.read_csv(fairdir + f"scen_{scenname}.csv").set_index("year")
        allruns = allruns[allruns.index <= 2100]
        runex = (allruns>exceedence).sum(axis=0)
        exceeds[scenname] = (runex > 0).sum() / len(runex)
    interesting[f"Temp exceeds {exceedence}"] = pd.Series(exceeds)
df = pd.DataFrame(interesting)
df.to_csv(outdir + "SummaryOfSenarios2100.csv")
