import matplotlib.pyplot as plt
import pandas as pd

path = "~/Documents/Pre-doc Data Task Practice/Data/cps_women_lfp.csv"

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
        rates = data.groupby(["year", group])["weighted_lfp"].sum() # compute sum of weighted lfp rates by year and group
    else: 
        yearly_weighted_sum = data.groupby("year")["wgt"].sum() # group by year and sum all the weights in each year
        data["lfp * wgt"] = data["lfp * wgt"] / data["year"].map(yearly_weighted_sum) # compute weighted lfp rates
        data = data.rename(columns={"lfp * wgt": "weighted_lfp"}) # rename column appropiately
        rates = data.groupby("year")["weighted_lfp"].sum() # compute sum of weighted lfp rates by year

    return rates

def q1(data): # How has female labor force participation evolved since 1994?
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

def q2(data, group=None): # Among women older than 25, which groups (race, age, income percentile, etc.) of people had the biggest changes in labor force participation since 1994?
    only_women_over_25 = data[(data["sex"] == "Female") & # filter for women older than 25 with lfp values that are not empty
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


if __name__ == "__main__": # run the function if the script is executed directly
    df = load_data(path)
    # q1(df) 
    q2(df, group="race") 
    q2(df, group="age")
    q2(df, group="income_quantiles")
    q2(df, group="education")
    q2(df, group="college")