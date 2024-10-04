import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Custom styling with theme configuration
st.set_page_config(page_title="Global Climate Analysis", page_icon="🌍", layout="wide")

user_name = "Priya Rai 24MAI0070"

# Function to generate synthetic temperature data
def generate_temperature_data(start_year, end_year):
    years = list(range(start_year, end_year + 1))
    temperatures = [14 + (0.02 * (year - start_year)) + np.random.uniform(-0.5, 0.5) for year in years]
    return pd.DataFrame({'Year': years, 'Temperature': temperatures})

# Function to generate synthetic CO2 emissions data
def generate_co2_data(start_year, end_year):
    years = list(range(start_year, end_year + 1))
    co2_emissions = [3200 + (15 * (year - start_year)) + np.random.uniform(-100, 100) for year in years]
    return pd.DataFrame({'Year': years, 'CO2 Emissions': co2_emissions})

# Function to plot with trend lines
def plot_with_trendlines(temperature_data, co2_data, year1, year2):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot temperature
    sns.lineplot(x=temperature_data['Year'], y=temperature_data['Temperature'], ax=ax, color='dodgerblue', label='Temperature (°C)')
    ax.fill_between(temperature_data['Year'], temperature_data['Temperature'], color='dodgerblue', alpha=0.3)

    # Plot CO2 emissions
    sns.lineplot(x=co2_data['Year'], y=co2_data['CO2 Emissions'], ax=ax, color='lightgreen', label='CO2 Emissions (Million Tonnes)')
    ax.fill_between(co2_data['Year'], co2_data['CO2 Emissions'], color='lightgreen', alpha=0.3)

    # Calculate and plot trendlines
    x_temp = temperature_data['Year'].values.reshape(-1, 1)
    y_temp = temperature_data['Temperature'].values
    model_temp = LinearRegression().fit(x_temp, y_temp)
    ax.plot(temperature_data['Year'], model_temp.predict(x_temp), color='blue', linestyle='--', label='Temperature Trend')

    x_co2 = co2_data['Year'].values.reshape(-1, 1)
    y_co2 = co2_data['CO2 Emissions'].values
    model_co2 = LinearRegression().fit(x_co2, y_co2)
    ax.plot(co2_data['Year'], model_co2.predict(x_co2), color='green', linestyle='--', label='CO2 Trend')

    # Highlight selected years
    ax.scatter(year1, temperature_data[temperature_data['Year'] == year1]['Temperature'].values[0], color='darkblue', s=100)
    ax.scatter(year2, co2_data[co2_data['Year'] == year2]['CO2 Emissions'].values[0], color='darkgreen', s=100)

    ax.set_title("Climate Data with Trends")
    ax.set_xlabel("Year")
    ax.set_ylabel("Values")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# Home Page
def home_page():
    st.title("Global Climate Analysis App")
    st.write(f"Created by {user_name}")
    st.write("""
    This app provides insights into climate change through synthetic data-driven global temperature trends and CO2 emissions. 
    Navigate through the sidebar to explore data trends, interact with year-based analysis, and download the data for further study.
    """)

# Temperature Trends Page
def temperature_trends_page():
    st.title("Global Temperature Trends")
    st.write(f"Created by {user_name}")

    # Generate synthetic data
    temperature_data = generate_temperature_data(1900, 2100)

    # Create figure and axes for Matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=temperature_data['Year'], y=temperature_data['Temperature'], ax=ax, color='dodgerblue', label='Temperature')
    ax.fill_between(temperature_data['Year'], temperature_data['Temperature'], color='dodgerblue', alpha=0.3)
    ax.set_title('Global Temperature Trends (1900-2100)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature (°C)')
    ax.grid(True)
    st.pyplot(fig)  # Pass the figure object explicitly

    # Display Dataframe in the app
    st.write(temperature_data)

    # Download Button for Temperature Data
    st.download_button(
        label="Download Temperature Data as CSV",
        data=temperature_data.to_csv(index=False).encode('utf-8'),
        file_name='temperature_data.csv',
        mime='text/csv',
    )

# CO2 Emissions Page
def co2_emissions_page():
    st.title("CO2 Emissions Analysis")
    st.write(f"Created by {user_name}")

    # Generate synthetic data
    co2_data = generate_co2_data(1900, 2100)

    # Create figure and axes for Matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=co2_data['Year'], y=co2_data['CO2 Emissions'], ax=ax, color='lightgreen')
    ax.set_title('CO2 Emissions (1900-2100)')
    ax.set_xlabel('Year')
    ax.set_ylabel('CO2 Emissions (Million Tonnes)')
    ax.grid(True)
    st.pyplot(fig)  # Pass the figure object explicitly

    # Display Dataframe in the app
    st.write(co2_data)

    # Download Button for CO2 Emissions Data
    st.download_button(
        label="Download CO2 Emissions Data as CSV",
        data=co2_data.to_csv(index=False).encode('utf-8'),
        file_name='co2_emissions_data.csv',
        mime='text/csv',
    )

# Enhanced Interactive Visualization Page
def interactive_visualization_page():
    st.title("Interactive Climate Data")
    st.write(f"Created by {user_name}")

    # Generate synthetic data
    temperature_data = generate_temperature_data(1900, 2100)
    co2_data = generate_co2_data(1900, 2100)

    # User inputs for year selection
    year1 = st.selectbox("Select First Year", temperature_data['Year'].tolist())
    year2 = st.selectbox("Select Second Year", temperature_data['Year'].tolist())

    # Filter data for the selected years
    temp1 = temperature_data[temperature_data['Year'] == year1]['Temperature'].values[0]
    co21 = co2_data[co2_data['Year'] == year1]['CO2 Emissions'].values[0]
    temp2 = temperature_data[temperature_data['Year'] == year2]['Temperature'].values[0]
    co22 = co2_data[co2_data['Year'] == year2]['CO2 Emissions'].values[0]

    # Display results
    st.write(f"In {year1}, the global temperature was {temp1:.2f} °C and CO2 emissions were {co21:.2f} million tonnes.")
    st.write(f"In {year2}, the global temperature was {temp2:.2f} °C and CO2 emissions were {co22:.2f} million tonnes.")

    # Calculate and display statistical summaries
    st.write("### Statistical Summary")
    selected_years_data = temperature_data[(temperature_data['Year'] == year1) | (temperature_data['Year'] == year2)]
    st.write(f"Mean Temperature: {selected_years_data['Temperature'].mean():.2f} °C")
    st.write(f"Median Temperature: {selected_years_data['Temperature'].median():.2f} °C")
    st.write(f"Temperature Range: {selected_years_data['Temperature'].min():.2f} °C to {selected_years_data['Temperature'].max():.2f} °C")

    selected_years_data_co2 = co2_data[(co2_data['Year'] == year1) | (co2_data['Year'] == year2)]
    st.write(f"Mean CO2 Emissions: {selected_years_data_co2['CO2 Emissions'].mean():.2f} million tonnes")
    st.write(f"Median CO2 Emissions: {selected_years_data_co2['CO2 Emissions'].median():.2f} million tonnes")
    st.write(f"CO2 Emissions Range: {selected_years_data_co2['CO2 Emissions'].min():.2f} million tonnes to {selected_years_data_co2['CO2 Emissions'].max():.2f} million tonnes")

    # Create a figure and plot the full data with highlights for the selected years
    plot_with_trendlines(temperature_data, co2_data, year1, year2)

    # Provide insights based on the selected years
    st.write(f"""
    ### Insights:
    - The global temperature in {year1} was **{temp1:.2f}°C** and in {year2} was **{temp2:.2f}°C**.
    - CO2 emissions in {year1} were **{co21:.2f} million tonnes** and in {year2} were **{co22:.2f} million tonnes**.
    - These changes reflect ongoing trends in climate change and industrial activity.
    """)

# Main function to control the navigation
def main():
    st.sidebar.title("Navigation")
    
    
    # Sidebar selection for navigation
    page = st.sidebar.selectbox("Go to", ["Home", "Temperature Trends", "CO2 Emissions Analysis", "Interactive Visualization"])
    
    if page == "Home":
        home_page()
    elif page == "Temperature Trends":
        temperature_trends_page()
    elif page == "CO2 Emissions Analysis":
        co2_emissions_page()
    elif page == "Interactive Visualization":
        interactive_visualization_page()

# Run the app
if __name__ == "__main__":
    main()
