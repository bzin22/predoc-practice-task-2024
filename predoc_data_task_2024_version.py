import matplotlib.pyplot as plt
import pandas as pd

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

################## PART 2 ##################

################## MAIN FUNCTION ##################
if __name__ == "__main__":
    df = load_data(path)

    # p1q1(df)    
    # p1q2(df, group="race") 
    # p1q2(df, group="age")
    # p1q2(df, group="income_quantiles")
    # p1q2(df, group="education")
    # p1q2(df, group="college")
    # p1q3(df, group="income")
    # p1q3(df, group="incss")
    