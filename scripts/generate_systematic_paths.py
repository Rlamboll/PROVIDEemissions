import numpy as np
import os
import pyam
import silicone.multiple_infillers as mi
import silicone.database_crunchers.quantile_rolling_windows as qrw

def make_paths(
        net_zeros, overshoots, mod_2030s, methane_levels, outdir, version, check_min_dif=True,
        make_scen_files=False
):
    sr15_em = pyam.IamDataFrame("../input/sr15_cleaned_harmed.csv").filter(
        region="World")
    complete_sr15 = pyam.IamDataFrame("../input/complete_sr15_emissions.csv").filter(
        region="World")
    scenario = "SSP2-45"
    model = "MESSAGE-GLOBIOM 1.0"
    sr15_em_int = sr15_em.interpolate([2025, 2035, 2045, 2055, 2065, 2075, 2085, 2095])
    co2tot = "Emissions|CO2"
    co2afolu = "Emissions|CO2|AFOLU"
    co2ei = "Emissions|CO2|Energy and Industrial Processes"
    basis_scen = sr15_em_int.filter(model=model, scenario=scenario,
                                    year=[2010, 2015, 2020])
    centyears = [2010, 2015] + list(np.arange(2020, 2101, 10))
    yeargap = 10
    max_grad = 5000
    # Sometimes we want to ensure that the difference in 2030-2040 does not exceed some
    # limit
    if check_min_dif:
        min_dif = -30902
    for mod_2030 in mod_2030s:
        for net_zero in net_zeros:
            for overshoot in overshoots:
                scen = basis_scen.filter(variable="*CO2").timeseries()
                scen[2030] = scen[2020] * mod_2030
                for year in range(2030 + yeargap, min(net_zero + 1, 2301), yeargap):
                    scen[year] = scen[2030] * (net_zero - year) / (net_zero - 2030)
                for year in range(net_zero + yeargap, 2301, yeargap):
                        scen[year] = max(
                            [overshoot, scen[year - yeargap].values - max_grad])
                if check_min_dif & ((scen[2040] - scen[2030] - min_dif).values[0] < 0):
                    continue
                scen = scen.reset_index()
                scen[
                    "scenario"] = f"2030fact{round(mod_2030, 3)}_nz{net_zero}_ov{overshoot}"
                try:
                    all_scens = all_scens.append(scen)
                except NameError:
                    all_scens = scen
    all_scens = pyam.IamDataFrame(all_scens)

    # Break down totals
    co2_infiller = mi.SplitCollectionWithRemainderEmissions(sr15_em)
    co2_breakdown = co2_infiller.infill_components(co2tot, [co2afolu], co2ei,
                                                   all_scens.filter(year=centyears))
    all_scens = all_scens.append(co2_breakdown)
    # Infill methane emissions
    scenarios = []
    methane_infiller = qrw.QuantileRollingWindows(sr15_em)
    for methane_level in methane_levels:
        methane_scen = all_scens.copy().data
        methane_scen["scenario"] = methane_scen["scenario"] + "_meth{}".format(
            round(methane_level, 3))
        methane_vals = methane_infiller.derive_relationship(
            "Emissions|CH4", ["Emissions|CO2"], quantile=methane_level
        )(pyam.IamDataFrame(methane_scen).filter(year=centyears))
        scenarios.append(pyam.IamDataFrame(methane_scen))
        scenarios.append(methane_vals)
    all_scens = pyam.concat(scenarios)
    # Infill the remaining emissions
    required_variables_list = [
        "Emissions|BC",
        "Emissions|CO",
        "Emissions|N2O",
        "Emissions|NH3",
        "Emissions|NOx",
        "Emissions|OC",
        "Emissions|Sulfur",
        "Emissions|VOC",
    ]
    other_infilled = mi.infill_all_required_variables(
        all_scens.filter(year=centyears),
        sr15_em,
        [co2tot],
        required_variables_list=required_variables_list
    )
    # Infill the f-gases (these are not harmonized). We use additional smoothing in this because there are fewer scenarios
    f_gases = [
        "Emissions|PFC|CF4",
        "Emissions|PFC|C2F6",
        "Emissions|PFC|C6F14",
        "Emissions|HFC|HFC134a",
        "Emissions|HFC|HFC143a",
        "Emissions|HFC|HFC227ea",
        "Emissions|HFC|HFC23",
        "Emissions|HFC|HFC32",
        "Emissions|HFC|HFC43-10",
        "Emissions|HFC|HFC245ca",
        "Emissions|HFC|HFC125",
        "Emissions|SF6",
    ]
    other_infilled = mi.infill_all_required_variables(other_infilled, complete_sr15,
                                                      [co2tot],
                                                      required_variables_list=f_gases,
                                                      nwindows=5, decay_length_factor=2)
    other_infilled = pyam.IamDataFrame(other_infilled)

    # Extend the scenarios to end of 2300
    last_vals = other_infilled.filter(year=2100).filter(variable=[co2tot, co2ei],
                                                        keep=False).data
    all_years_extension = []
    extend_years = [y for y in all_scens.year if y > 2100]
    for year in extend_years:
        year_val = last_vals.copy()
        year_val["year"] = year
        all_years_extension.append(year_val)
    extensions = pyam.concat(all_years_extension)
    extensions2 = extensions.append(all_scens.filter(year=extend_years))
    extend_eni = extensions2.subtract(co2tot, co2afolu, co2ei, ignore_units=True)
    extend_eni = extend_eni.rename(unit={"unknown": "Mt CO2/yr"})
    other_infilled = pyam.concat([other_infilled, extensions2, extend_eni])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    csv_em_file = outdir + "scenarios_{}.csv"
    scen_file = outdir + "scen_{}"
    other_infilled.to_csv(csv_em_file.format(version))
    if make_scen_files:
        for scenario in other_infilled.scenario:
          utils.construct_scen_file(other_infilled.filter(scenario=scenario), scen_file.format(scenario))

