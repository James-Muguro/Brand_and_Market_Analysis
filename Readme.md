# Feasibility Evaluation Model for Solar-Battery Integration

## Project Description
This project delves into the practicality of augmenting Naomi’s current solar power system with a battery. By analyzing Naomi’s hourly electricity data from 2020, the model estimates potential solar generation, battery storage, cost savings, and long-term returns. The analysis considers two future electricity price scenarios to provide a comprehensive view of the potential benefits of this sustainable energy solution.

## Purpose of the Model
The model’s primary objective is to offer valuable insights into the potential cost savings and financial viability of such an integration. It serves as a tool for understanding the economic implications of adopting a battery system alongside existing solar power infrastructure.

## Data Checks and Results
The dataset undergoes rigorous checks to ensure data completeness and reliability. The following steps are performed:

1. **Verification of Data Completeness:**
   - Ensure all necessary fields are populated.
   - Handle missing values appropriately.

2. **Outlier Investigation:**
   - Identify and address outliers using boxplots.

## Assumptions
The model operates under the following assumptions:

- **Electricity Cost:** The cost of electricity per kWh is assumed to be $0.17.
- **Battery Charge Level:** The maximum battery charge level is set at 100 kWh.

## Data Understanding
To gain a comprehensive understanding of the dataset, we undertake the following:

1. **Data Types and Distributions:**
   - Explore data types and distributions for effective analysis.

2. **Visualization:**
   - Utilize time-series line plots and histograms for visualization.

## Modeling Steps and Checks
The model follows a systematic approach with clear steps and checks:

1. **Average Solar Electricity Generation and Electricity Usage per Hour:**
   - Calculate and visualize hourly averages.

2. **Outlier Investigation:**
   - Utilize boxplots to identify and address outliers.

3. **Electricity Bought and Excess Solar Electricity Generated:**
   - Calculate hourly electricity bought and excess solar electricity generated.

4. **Cumulative Battery Charge Modeling:**
   - Model the cumulative charge level of the battery over time.

5. **Electricity Bought with Battery Installed:**
   - Calculate electricity bought when a battery is installed.

6. **Savings Calculation:**
   - Determine the savings from installing a battery.

7. **Monthly Tabulation and Charting:**
   - Tabulate and visualize monthly solar generation, electricity usage, and purchased electricity.

## Long-Term Projections
To project future savings, we consider two scenarios of electricity price changes over 20 years.

## Further Checks
- **Internal Rate of Return (IRR) Calculation:**
  - Calculate IRR for different scenarios.

- **Additional Metrics:**
  - Evaluate extra electricity met, implied savings, electricity price with inflation, NPV, and IRR.

## Conclusion
This project is dedicated to offering valuable insights into the financial implications of integrating a battery into solar electricity generation. Leveraging the capabilities of Python and key libraries such as NumPy, Pandas, ipywidgets, tkinter, matplotlib, and seaborn, the project benefits from the efficient array manipulation tools provided by the NumPy community and the versatile visualization tools from the developers of matplotlib and seaborn. Throughout the development of this project, online tutorials and responses from the Stack Overflow community have played a crucial role, providing invaluable guidance and support.
