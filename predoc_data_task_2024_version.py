import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

path = "~/predoc-practice-task-2024/Data/cps_women_lfp.csv"

################## HELPER FUNCTIONS ##################
def load_data(file_path):
    return pd.read_csv(file_path)

def yearly_weighted_lfp(data, group=None): # rate of lfp per year = sum(wgt_i / sum(wgt) * lfp_i)
    mapping = {
            "In labor force": 1,
            "Not in labor force": 0
    }
    data["lfp"] = data["lfp"].map(mapping).astype(int) # change lfp from str to binary classifier

    # add a new column with lfp * weight to the filtered dataset
    lfp_x_wgt = data["lfp"] * data["wgt"]
    data["lfp * wgt"] = lfp_x_wgt

    if group is not None: 
        data["lfp * wgt"] = data["lfp * wgt"] / data.groupby(["year", group])["wgt"].transform(sum) # compute weighted lfp rates
        data = data.rename(columns={"lfp * wgt": "weighted_lfp"}) # rename column appropiately
        # print(data[data["year"] == 2022, data["covid_telework"]])
        rates = data.groupby(["year", group])["weighted_lfp"].sum() # compute sum of weighted lfp rates by year and group
    else: 
        yearly_weighted_sum = data.groupby("year")["wgt"].sum() # group by year and sum all the weights in each year
        data["lfp * wgt"] = data["lfp * wgt"] / data["year"].map(yearly_weighted_sum) # compute weighted lfp rates
        data = data.rename(columns={"lfp * wgt": "weighted_lfp"}) # rename column appropiately
        rates = data.groupby("year")["weighted_lfp"].sum() # compute sum of weighted lfp rates by year

    return rates

def weighted_median(data, group): 
    ascending_order = data.sort_values(["year", group])[["year", group, "wgt"]] # sort values in ascending order for each year
    target_index = ascending_order.groupby("year")["wgt"].transform(sum)  / 2 # compute the target index for median
    running_total = ascending_order.groupby("year")["wgt"].cumsum() # compute the running total of weights
    weighted_median = ascending_order[running_total >= target_index].groupby("year").first()[group] # compute weighted median by year
    
    return weighted_median

def weighted_mean(data, group):
    weighted_mean = data[group] * data["wgt"] / data.groupby("year")["wgt"].transform("sum") # compute weighted mean by year
    data["weighted_mean"] = weighted_mean # add a new column
    
    return data.groupby("year")["weighted_mean"].sum() # return DataFrame with each year's weighted mean

def weighted_rate(rows, flag):
        return (rows[flag] * rows["wgt"]).groupby(rows["year"]).sum() / rows.groupby("year")["wgt"].sum()

################## PART 1 ##################
def p1q1(data): # How has female labor force participation evolved since 1994?
    only_women = data[(data["sex"] == "Female") & (data["lfp"].notna())].copy() # filter for women only and lfp values that are not empty
    only_women_working_age = only_women[only_women["age"].isin(["25-34", "35-44", "45-54"])].copy() # filter for women of working age (25-54)

    rates = yearly_weighted_lfp(only_women) # compute weighted lfp rates for all women
    rates_working_age = yearly_weighted_lfp(only_women_working_age) # compute weighted lfp rates for women of working age

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(rates.index, rates.values, label="Weighted LFP (all women) vs Year", color="blue", marker="o")
    plt.plot(rates_working_age.index, rates_working_age.values, label="Weighted LFP (Working Age) vs Year", color="orange", marker="o")
    plt.xlabel("Year")
    plt.ylabel("Annual LFP")
    plt.legend()
    plt.title("U.S. Women's LFP vs Year")
    plt.grid(True)
    plt.show()

def p1q2(data, group=None): # Among women 25 and over, which groups (race, age, income percentile, etc.) of people had the biggest changes in labor force participation since 1994?
    only_women_over_25 = data[(data["sex"] == "Female") & # filter for women 25 and over with lfp values that are not empty
                              (data["lfp"].notna()) &
                              (data["age"].isin(["25-34", "35-44", "45-54", "55-64", "65-74", "75+"]))].copy() 

    rates = yearly_weighted_lfp(only_women_over_25, group) # compute weighted lfp rates for all women
    rates_by_group = rates.unstack(level=group) # unstack the multi-index series to a data table for plotting

    # plot
    plt.figure(figsize=(8, 5))
    for group_value in rates_by_group.columns:
        plt.plot(rates_by_group.index, rates_by_group[group_value], marker="o", label=group_value)
    plt.xlabel("Year")
    plt.ylabel("Annual LFP")
    plt.legend()
    plt.title(f"Weighted LFP by {group.capitalize()} vs Year")
    plt.grid(True)
    plt.show()

def p1q3(data, group=None): # examine trends among women 25 and over for wage & salary income, social insurance income, and education attainment
    only_women_over_25 = data[(data["sex"] == "Female") & # filter for women 25
                              (data["age"].isin(["25-34", "35-44", "45-54", "55-64", "65-74", "75+"]))].copy() 
    
    filtered_data = only_women_over_25[only_women_over_25[group].notna()] # filter for women with group values that are not empty
    
    if is_numeric_dtype(filtered_data[group]):
        wmed = weighted_median(filtered_data, group) # compute weighted median 
        wmean = weighted_mean(filtered_data, group) # compute weighted mean 

        # plot
        plt.figure(figsize=(8, 5))
        plt.plot(wmean.index, wmean.values, marker="o", label=f"Mean Weighted {group.capitalize()}", color="blue")
        plt.plot(wmed.index, wmed.values, marker="o", label=f"Weighted Median {group.capitalize()}", color="orange")
        plt.xlabel("Year")
        plt.ylabel("Annual " + group.capitalize())
        plt.legend()
        plt.title(f"{group.capitalize()} vs Year")
        plt.grid(True)
        plt.show()
    else: 
        weights_by_subgroup = filtered_data.groupby(["year", group])["wgt"].sum().unstack(level=group) # compute sum of weights by year and group
        weighted_percent_by_subgroup = weights_by_subgroup.div(weights_by_subgroup.sum(axis=1), axis=0) # compute weighted percent by year and group

        # plot
        plt.figure(figsize=(8, 5))
        for subgroup in weighted_percent_by_subgroup.columns:
            plt.plot(weighted_percent_by_subgroup.index, weighted_percent_by_subgroup[subgroup], marker="o", label=f"{subgroup.capitalize()}")
        plt.xlabel("Year")
        plt.ylabel("Weighted Percent")
        plt.legend()
        plt.title(f"Weighted Percent for highest level of {group.capitalize()} vs Year")
        plt.grid(True)
        plt.show()

def p1q6(data, group=None): # Create an alternative measure of labor force participation that excludes individuals from the labor force if they are self-employed in their main job (lfp = 0 if self-employed in main job). Using the new measure, describe how labor force participation for college-educated and not college-educated women has evolved since 1994.
    only_women_over_25 = data[(data["sex"] == "Female") & # filter for women 25 and over with lfp values that are not empty
                                      (data["lfp"].notna()) &
                                      (data["age"].isin(["25-34", "35-44", "45-54", "55-64", "65-74", "75+"]))].copy() 

    # if self-employed  == 1, then set lfp = 0
    only_women_over_25.loc[only_women_over_25["self_employed"] == "Self-employed", "lfp"] = "Not in labor force"

    rates = yearly_weighted_lfp(only_women_over_25, group) # compute weighted lfp rates for all women
    rates_by_group = rates.unstack(level=group) # unstack the multi-index series to a data table for plotting

    # plot
    plt.figure(figsize=(8, 5))
    for group_value in rates_by_group.columns:
        plt.plot(rates_by_group.index, rates_by_group[group_value], marker="o", label=group_value)
    plt.xlabel("Year")
    plt.ylabel("Annual LFP")
    plt.legend()
    plt.title(f"Weighted LFP by {group.capitalize()} vs Year")
    plt.grid(True)
    plt.show()

################## PART 2 ##################
def p2q1(data): # how have wages, employment, and labor force participation changed for women who had telework from 2020-2024 and women who did not?
    # only_women_2020_2024 = data[(data["sex"] == "Female") & 
    #                             data["covid_telework"].notna() &
    #                             data["wageinc_quantiles"].notna() &
    #                             data["telework_now"].notna() &
    #                             data["year"].isin([2020, 2021, 2022, 2023, 2024])].copy() # filter for women from 2020-2024

    # only_women_2020_2024["telework_status"] = "Not enough information" # create a new column to indicate telework status
    # only_women_2020_2024.loc[only_women_2020_2024["telework_now"] == "Had telework in the past week", "telework_status"] = "Teleworked between 2020-2024"
    # only_women_2020_2024.loc[only_women_2020_2024["covid_telework"] == "Telework from 2021-2022 due to COVID", "telework_status"] = "Teleworked between 2020-2024"

    women = data[(data["sex"] == "Female") & data["year"].isin([2020, 2021, 2022, 2023, 2024])]
    only_women_2020_2024 = women[(women["covid_telework"].notna() | women["telework_now"].notna()) &
                                 (women["employed"] == "Employed")].copy()

    only_women_2020_2024["telework_status"] = "Did not telework"
    only_women_2020_2024.loc[only_women_2020_2024["telework_now"] == "Had telework in the past week", "telework_status"] = "Teleworked between 2020-2024"
    only_women_2020_2024.loc[only_women_2020_2024["covid_telework"] == "Telework from 2021-2022 due to COVID", "telework_status"] = "Teleworked between 2020-2024"
    only_women_2020_2024["teleworked"] = (only_women_2020_2024["telework_status"] == "Teleworked between 2020-2024").astype(int)

    # income is on different survey rows than telework answers, so attach it by person ID and year
    incomes = women[women["income"].notna()].drop_duplicates(["cpsidp", "year"])[["cpsidp", "year", "income"]]
    only_women_2020_2024 = only_women_2020_2024.drop(columns="income").merge(incomes, on=["cpsidp", "year"], how="inner")

    # plot
    plt.figure(figsize=(9, 9))
    for status, status_rows in only_women_2020_2024.groupby("telework_status"):
        wmean = weighted_mean(status_rows.copy(), "income")
        plt.plot(wmean.index, wmean.values, marker="o", label=f"{status} (weighted mean)")
    plt.xlabel("Year")
    plt.ylabel("Annual income")
    plt.legend()
    plt.title("Mean Income for Teleworking vs Non-Teleworking Women by Year")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(9, 9))
    for status, status_rows in only_women_2020_2024.groupby("telework_status"):
        wmed = weighted_median(status_rows, "income")
        plt.plot(wmed.index, wmed.values, marker="o", linestyle="--", label=f"{status} (weighted median)")
    plt.xlabel("Year")
    plt.ylabel("Annual income")
    plt.legend()
    plt.title("Median Income for Teleworking vs Non-Teleworking Women by Year")
    plt.grid(True)
    plt.show()

def p2q2(data): # For which groups of women older than 25 was telework due to the pandemic most common in 2021?
    def plot_weighted_percent_by_group(data, group=None):
        filtered_data = data[(data[group].notna())].copy()

        weights_by_group = filtered_data.groupby(group)["wgt"].sum() # compute sum of weights by group
        weighted_percent_by_group = weights_by_group / weights_by_group.sum() # compute weighted percent by group

        # plot the weighted percent of women who teleworked due to the pandemic in 2021, grouped by education level
        plt.figure(figsize=(9, 9))
        plt.bar(weighted_percent_by_group.index, weighted_percent_by_group.values, color="skyblue")
        plt.xlabel(f"{group.capitalize()}")
        plt.ylabel("Weighted Percent")
        plt.legend()
        plt.title(f"Percentage of Women who Teleworked due to Pandemic by {group.capitalize()}")
        plt.grid(True)
        plt.show()
    
    women = data[(data["sex"] == 'Female') & # women 25 or older in 2021 with telework due to the pandemic
                  (data["age"].isin(["25-34", "35-44", "45-54", "55-64", "65-74", "75+"])) &
                  (data["covid_telework"] == "Telework from 2021-2022 due to COVID") &
                  (data["year"] == 2021)].copy()

    plot_weighted_percent_by_group(women, group="education")
    plot_weighted_percent_by_group(women, group="race")
    plot_weighted_percent_by_group(women, group="age")

def p2q3(data): # Predict what trends in wages, employment, and labor force participation for college-educated women from 2020 to 2024 would have looked like if telework was not an option.
    women = data[(data["sex"] == "Female") & (data["age"] != "< 25") & data["age"].notna() &
             data["college"].notna() & (data["year"] >= 1994)].copy()

    outcomes = {}
    for college, rows in women.groupby("college"):
        in_pop = rows[rows["lfp"].notna()].copy()
        in_pop["in_lf"] = (in_pop["lfp"] == "In labor force").astype(int)
        in_pop["is_employed"] = (in_pop["employed"] == "Employed").astype(int)
        earners = rows[rows["income"] > 0]
        outcomes[college] = pd.DataFrame({
            "LFP": weighted_rate(in_pop, "in_lf"),
            "Employment rate": weighted_rate(in_pop, "is_employed"),
            "Median wage income": weighted_median(earners, "income"),
        })

    college, non_college = outcomes["Has college degree"], outcomes["No college degree"]
    fit_years = np.arange(2010, 2020)  # trend is fit on the recent decade only; charts still show 1994 onward
    rows_out = []
    for outcome in college.columns:
        c, n = college[outcome], non_college[outcome]
        if outcome == "Median wage income":
            # ratio version: inflation cancels because both groups face the same prices
            gap = np.log(c / n)
            cf_simple = c[2019] * n / n[2019]
            slope, intercept = np.polyfit(fit_years, gap.loc[fit_years], 1)
            cf_trend = n * np.exp(intercept + slope * n.index)
            print(f"{outcome}: pre-2020 log gap slope {slope*100:+.2f}% per year")
        else:
            gap = c - n
            cf_simple = c[2019] + (n - n[2019])
            slope, intercept = np.polyfit(fit_years, gap.loc[fit_years], 1)
            cf_trend = n + intercept + slope * n.index
            print(f"{outcome}: pre-2020 gap slope {slope*100:+.2f} points per year, gap 1994 {gap[1994]*100:.1f} -> 2019 {gap[2019]*100:.1f}")
        for year in range(2019, 2025):
            if year in c.index and not pd.isna(c.get(year)):
                rows_out.append([outcome, year, c[year], n[year], cf_simple[year], cf_trend[year]])

        fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 10), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
        top.plot(c.index, c.values, marker="o", label="College, actual")
        top.plot(n.index, n.values, marker="o", label="Non-college, actual")
        post = cf_simple.loc[2019:]
        top.plot(post.index, post.values, linestyle="--", marker="x", label="College, no-telework counterfactual (simple)")
        post_t = cf_trend.loc[2019:]
        top.plot(post_t.index, post_t.values, linestyle=":", marker="x", label="College, no-telework counterfactual (trend-adjusted)")
        top.axvline(2019.5, color="grey", linewidth=0.8)
        top.set_ylabel(outcome); top.legend(); top.grid(True)
        top.set_title(f"{outcome}, women 25+, by college degree")

        # the gap the trend line is fit to, with the fitted line extended past 2019
        if outcome == "Median wage income":
            # show dollars; the fitted line is the log fit converted back to dollars (trend counterfactual minus non-college)
            gap_label = "College - non-college, dollars"
            bottom.plot(c.index, (c - n).values, marker="o", color="black", label="Actual gap")
            bottom.plot(n.loc[2010:].index, (cf_trend - n).loc[2010:].values, linestyle=":", color="tab:red", label="Fitted 2010-2019 line (log fit, in dollars)")
        else:
            gap_label = "College - non-college, points"
            bottom.plot(gap.index, gap.values * 100, marker="o", color="black", label="Actual gap")
            bottom.plot(gap.loc[2010:].index, (intercept + slope * gap.loc[2010:].index) * 100, linestyle=":", color="tab:red", label="Fitted 2010-2019 line")
        bottom.axvline(2019.5, color="grey", linewidth=0.8)
        bottom.set_xlabel("Year"); bottom.set_ylabel(gap_label); bottom.legend(); bottom.grid(True)
        plt.show()
        fig.tight_layout()
        # fig.savefig(f"/tmp/p2q3_did/from1994_{outcome.replace(' ', '_')}.png"); plt.close(fig)

    table = pd.DataFrame(rows_out, columns=["outcome", "year", "college", "non_college", "cf_simple", "cf_trend"])
    table["effect_simple"] = table["college"] - table["cf_simple"]
    table["effect_trend"] = table["college"] - table["cf_trend"]
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))

    # noise check: how far does the pre-2020 gap wander from its own trend in normal years?
    for outcome in college.columns:
        c, n = college[outcome].loc[fit_years], non_college[outcome].loc[fit_years]
        gap = np.log(c / n) if outcome == "Median wage income" else (c - n)
        resid = gap - np.polyval(np.polyfit(fit_years, gap, 1), fit_years)
        unit = "% (log)" if outcome == "Median wage income" else " points"
        print(f"{outcome}: pre-2020 year-to-year noise (SD of gap around trend) {resid.std()*100:.2f}{unit}, max {abs(resid).max()*100:.2f}")

################## MAIN FUNCTION ##################
if __name__ == "__main__":
    df = load_data(path)

    # p1q1(df) # figure 1
    # p1q2(df, group="race") # figure 1
    # p1q2(df, group="age") # figure 2
    # p1q2(df, group="income_quantiles") # figure 3
    # p1q2(df, group="education") # figure 4
    # p1q2(df, group="college") # figure 5
    # p1q3(df, group="income") # 3a figure 1
    # p1q3(df, group="incss") # 3b figure 1
    # p1q3(df, group="education") # 3c figure 1
    # p1q6(df, group="college") # figure 1
    # p2q1(df) # figure 1
    # p2q2(df) # figure 1， 2， 3     
    # p2q3(df) # figure 1, 2, 3