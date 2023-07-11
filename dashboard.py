import pandas as pd
import glob
import re
import plotly.graph_objects as go
import numpy as np
import pycountry
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import Dash, dcc, html


# Specify the path to the directory containing your CSV files
path = 'C:\\Users\\Eston\\OneDrive\\Desktop\\project datasets to use\\waste scan\\wastebase\\*.csv'

# Get a list of all CSV file paths in the directory
csv_files = glob.glob(path)

# Initialize an empty list to store the individual dataframes
dfs = []

# Iterate over each CSV file, read it as a dataframe, and append it to the list
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)

# Merge the dataframes using the pd.concat() function
merged_df = pd.concat(dfs, ignore_index=True)


# Fill missing values in the manufacturer_country column with 'NA' for Namibia
merged_df['manufacturer_country'].fillna('NA', inplace=True)


# Function to update values
def update_product_size(value):
    # Check if the value doesn't have a letter suffix
    if re.match(r'^\d+\.?\d*$', value):
        size = float(value)
        # Assign ML suffix if the value is greater than 50
        if size > 50:
            return str(size) + 'ML'
        # Assign L suffix if the value is less than 21
        if size < 21:
            return str(size) + 'L'
    return value

# Apply the function to update product_size values
merged_df['product_size'] = merged_df['product_size'].apply(update_product_size)


# Function to update values
def update_product_size(value):
    # Check if the value doesn't have a letter suffix
    if re.match(r'^\d+\.?\d*$', value):
        size = float(value)
        # Assign ML suffix if the value is greater than 50
        if size > 50:
            return str(size) + 'ML'
        # Assign L suffix if the value is less than 21
        if size < 21:
            return str(size) + 'L'
    return value

# Apply the function to update product_size values
merged_df['product_size'] = merged_df['product_size'].apply(update_product_size)

# Function to update values
def update_product_size(value):
    # Check if the value has the 'L' suffix and is greater than 100
    if value.upper().endswith('L'):
        try:
            size = float(value[:-1])  # Remove 'L' suffix and convert to float
            # Convert to 'ML' if the value is greater than 100
            if size > 50:
                return str(size) + 'ML'
        except ValueError:
            pass
    return value

# Apply the function to update product_size values
merged_df['product_size'] = merged_df['product_size'].apply(update_product_size)

# Step 1: Group the data by brand_name
grouped_by_brand = merged_df.groupby('brand_name')

# Step 2: Group the data by brand_name and product_size
grouped_by_brand_size = merged_df.groupby(['brand_name', 'product_size'])

# Step 3: Find the most frequent bottle_weight value within each group and replace missing values
for (brand_name, product_size), group in grouped_by_brand_size:
    mode_weights = group['bottle_weight'].mode()
    if not mode_weights.empty:
        mode_weight = mode_weights.iloc[0]  # Get the most frequent bottle_weight value
        merged_df.loc[group.index, 'bottle_weight'] = merged_df.loc[group.index, 'bottle_weight'].fillna(mode_weight)

        # Group the data by product_size and calculate the mean bottle weight for each group
grouped_by_size = merged_df.groupby('product_size')
mean_weights = grouped_by_size['bottle_weight'].mean()


# Fill missing values with group means
# merged_df['bottle_weight'] = merged_df.groupby('product_size')['bottle_weight'].apply(lambda x: x.fillna(x.mean()))
merged_df['bottle_weight'] = merged_df.groupby('product_size')['bottle_weight'].transform(lambda x: x.fillna(x.mean()))

# Calculate the mean of the non-missing values in the bottle_weight column
mean_weight = merged_df['bottle_weight'].mean(skipna=True)

# Fill missing values in the bottle_weight column with the mean
merged_df['bottle_weight'].fillna(mean_weight, inplace=True)

# Calculate the mean of non-zero values in the bottle_weight column
mean_weight = merged_df[merged_df['bottle_weight'] > 0]['bottle_weight'].mean()

# Replace 0 values with the mean
merged_df.loc[merged_df['bottle_weight'] == 0, 'bottle_weight'] = mean_weight

merged_df['total_bottle_weight'] = merged_df['bottle_count'] * merged_df['bottle_weight'] /1000

# Group the data by scan_country and calculate the sum of bottle_count and total_bottle_weight
grouped_df = merged_df.groupby('scan_country').agg({'bottle_count': 'sum', 'total_bottle_weight': 'sum'}).reset_index()

# Create a new DataFrame called mapping_df
mapping_df = pd.DataFrame(grouped_df)

# Load the population data from the other file into a new DataFrame called population_df
population_df = pd.read_csv('World Population.csv')

# Merge the population data into the mapping_df DataFrame based on matching values
merged_mapping_df = mapping_df.merge(population_df[['CCA3', 'Name', '2022']], 
                                     left_on='scan_country', 
                                     right_on='CCA3', 
                                     how='left')

# Rename the columns
merged_mapping_df.rename(columns={'Name': 'country_name', '2022': 'population'}, inplace=True)

# Drop the 'CCA3' column from the merged_mapping_df DataFrame
merged_mapping_df.drop('CCA3', axis=1, inplace=True)

# Calculate the ratio of bottle_count to population
merged_mapping_df['bottle_count_per_capita'] = merged_mapping_df['bottle_count'] / (merged_mapping_df['population'] * 1000)



# Map two-letter country codes to ISO alpha-3 codes
merged_mapping_df['iso_alpha3'] = merged_mapping_df['scan_country'].apply(
    lambda x: pycountry.countries.get(alpha_2=x).alpha_3 if pycountry.countries.get(alpha_2=x) else None
)


merged_mapping_df = merged_mapping_df.drop(['scan_country', 'population'], axis=1)

# Sort the data by bottle_count_per_capita in descending order
bottle_count_per_capita_bar_df = merged_mapping_df.sort_values('bottle_count_per_capita', ascending=False)

# Sort the data by total_bottle_weight in descending order
total_bottle_weight_bar_df = merged_mapping_df.sort_values('total_bottle_weight', ascending=False)


########## 2. Manufacturer and contribution to Environmental Pollution #####################3

# Merge merged_df with population_df to get country names
manufacturer_df = merged_df.merge(population_df[['CCA3', 'Name']], left_on='scan_country', right_on='CCA3', how='left')

# Rename the 'Name' column to 'country_name'
manufacturer_df.rename(columns={'Name': 'country_name'}, inplace=True)

# Drop the 'CCA3' column as it is no longer needed
manufacturer_df.drop('CCA3', axis=1, inplace=True)

manufacturer_grouped = manufacturer_df.groupby('manufacturer_name').agg({'bottle_count': 'sum', 'total_bottle_weight': 'sum'})

############################Top 10 brands with most plastic bottles found in Environment###################
# Sort the manufacturer_grouped DataFrame by bottle_count in descending order
top_10_manufacturers_count = manufacturer_grouped.sort_values('bottle_count', ascending=False).head(10)

# Sort the manufacturer_grouped DataFrame by total_bottle_weight in descending order
top_10_manufacturers_weight = manufacturer_grouped.sort_values('total_bottle_weight', ascending=False).head(10)

# Create a list of unique country names from the manufacturer_df
country_names = manufacturer_df['country_name'].unique()

##################################### DASHBOARD CODE  ####################################


# Create the Dash app
app = dash.Dash(__name__)

# Set suppress_callback_exceptions=True
app.config.suppress_callback_exceptions = True

# Define the layout of the app
app.layout = html.Div(
    style={'width': '100%', 'display': 'flex'},
    children=[
        html.Div(
            style={'width': '9%', 'height': '100vh', 'overflow-y': 'auto'},
            children=[
                html.Button("Overview", id="button-overview", style={'width': '100%', 'height': '20px', 'margin-bottom': '3px'}),
                html.Button("Manufacturers", id="button-manufacturers", style={'width': '100%', 'height': '20px', 'margin-bottom': '3px'}),
                html.Button("Countries", id="button-countries", style={'width': '100%', 'height': '20px', 'margin-bottom': '3px'})
            ]
        ),
        html.Div(
            style={'width': '100%', 'padding': '20px'},
            children=[
                html.H1("Plastic Bottle Pollution Analysis Dashboard", style={'text-align': 'center', 'text-decoration': 'underline'}),
                html.Div(id="page-content")
            ]
        )
    ]
)


# Callbacks to switch between pages
@app.callback(
    Output("page-content", "children"),
    Input("button-overview", "n_clicks"),
    Input("button-manufacturers", "n_clicks"),
    Input("button-countries", "n_clicks")
)
def render_page(overview_clicks, manufacturers_clicks, countries_clicks):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'button-overview':
        # Create the content for the Overview page
        return html.Div([
            html.H2("Overview of Bottle Pollution By:", style={'text-align': 'center'}),

            # Buttons to toggle between the maps
            html.Div([
                html.Button("Bottle Count / Bottle Count per Capita", id="button-bottle-count", n_clicks=0, style={'margin-right': '70px'}),
                html.Button("Bottle Weight", id="button-bottle-weight", n_clicks=0)
            ], style={'display': 'flex', 'justify-content': 'center', 'margin-bottom': '20px'}),

            # Container for the map
            html.Div(id="map-container")
        ])


    elif button_id == 'button-manufacturers':
        # Create the content for the Manufacturers page
        return html.Div([
            html.H2(
                children=[
                    "Top 10 Manufacturers with the most Plastic ",
                    dcc.Dropdown(
                        id="dropdown-metric",
                        options=[
                            {'label': 'Bottles', 'value': 'bottles'},
                            {'label': 'Bottle Weight', 'value': 'bottle_weight'}
                        ],
                        value='bottles',
                        clearable=False,
                        style={'display': 'inline-block', 'width': '170px', 'vertical-align': 'middle', 'text-align': 'center'}
                    ),
                    " found in the Environment"
                ],
                style={'display': 'flex', 'align-items': 'center'}
            ),

            # Container for the bar graph
            html.Div(id="bar-container", style={'width': '100%', 'margin-top': '20px'})
        ])


    elif button_id == 'button-countries':
        # Create the content for the Countries page
        return html.Div([
            html.H2("Countries Pollution by Manufacturers Summary"),
            # Add the content for the countries page here
            html.Div([
                dcc.Dropdown(
                    id='country-dropdown',
                    options=[{'label': country, 'value': country} for country in country_names],
                    placeholder="Select a country",
                    clearable=False,
                ),
            ], style={'width': '200px', 'margin-bottom': '20px'}),
            html.Div(id='manufacturer-output')
        ])

    else:
        # Default page or welcome message
        return html.Div([
            html.Img(src="C:\\Users\\Eston\\OneDrive\\Desktop\\project datasets to use\\waste scan\\cover.png", style={'width': '400px', 'height': 'auto', 'margin-top': '20px'})

        ])



# Callback to update the map based on the selected button
@app.callback(
    Output("map-container", "children"),
    Input("button-bottle-count", "n_clicks"),
    Input("button-bottle-weight", "n_clicks")
)
def update_map(button_count, button_weight):
    if button_count > button_weight:
        # Create the Bottle Count / Bottle Count per Capita map
        colors = {
            'black': [merged_mapping_df['iso_alpha3'].iloc[merged_mapping_df['bottle_count_per_capita'].idxmax()]],
            'red': merged_mapping_df.nlargest(6, 'bottle_count_per_capita')['iso_alpha3'].tolist(),
            'green': merged_mapping_df.nlargest(11, 'bottle_count_per_capita')['iso_alpha3'].tolist()[6:],
            'skyblue': merged_mapping_df.nsmallest(len(merged_mapping_df) - 17, 'bottle_count_per_capita')['iso_alpha3'].tolist()
        }

        fig = go.Figure(data=go.Choropleth(
            locations=merged_mapping_df['iso_alpha3'],
            text=merged_mapping_df['country_name'],
            z=merged_mapping_df['bottle_count_per_capita'],
            colorscale=['green'] + ['yellow'] + ['orange'] * 5 + ['red'] * (len(merged_mapping_df) - 17),
            autocolorscale=False,
            marker_line_color='white',
            marker_line_width=0.5,
            colorbar=dict(
                title='BottleCount<br>PerCapita<br>(Per Person)<br>Scale',
                x=0,
                xanchor='left',
                len=0.9,
                y=0.9,
                yanchor='top',
                thickness=20,
                tickformat='.2f'
            ),
            zmin=0,
            zmax=np.percentile(merged_mapping_df['bottle_count_per_capita'], 95)
        ))

        fig.update_layout(
            title='Plastic Bottle Count and Bottle Count Per Capita for Each Country',
            geo=dict(       
                showframe=True,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            margin=dict(l=30, r=30, t=30, b=30)
        )

        for color, countries in colors.items():
            indices = [i for i, country in enumerate(merged_mapping_df['iso_alpha3']) if country in countries]
            fig.update_traces(marker=dict(color=color), selector=dict(indices=indices))

        # Update hover text and hover template
        fig.update_traces(hovertext=merged_mapping_df['country_name'] + '<br>Bottle Count Per Capita: ' +
                                merged_mapping_df['bottle_count_per_capita'].apply(lambda x: f'{x:.4f}') +
                                '<br>Bottle Count: ' + merged_mapping_df['bottle_count'].astype(str),
                        hovertemplate='%{hovertext}')


        # Create the bar plot
        bar = go.Figure(data=go.Bar(
            x=bottle_count_per_capita_bar_df['bottle_count_per_capita'],
            y=bottle_count_per_capita_bar_df['country_name'],
            orientation='h'
        ))

        # Customize the layout
        bar.update_layout(
            title='Bottle Count per Capita by Country',
            xaxis_title='Bottle Count Per Capita',
            yaxis_title='Country',
            bargap=0.1,
            bargroupgap=0.05,
            height=600,
            width=800
        )

        return html.Div([
            dcc.Graph(figure=fig),
            dcc.Graph(figure=bar)
        ])


    else:
        # Create the Bottle Weight map
        colors = {
            'black': [merged_mapping_df['iso_alpha3'].iloc[merged_mapping_df['total_bottle_weight'].idxmax()]],
            'red': merged_mapping_df.nlargest(6, 'total_bottle_weight')['iso_alpha3'].tolist(),
            'green': merged_mapping_df.nlargest(11, 'total_bottle_weight')['iso_alpha3'].tolist()[6:],
            'skyblue': merged_mapping_df.nsmallest(len(merged_mapping_df) - 17, 'total_bottle_weight')['iso_alpha3'].tolist()
        }

        fig = go.Figure(data=go.Choropleth(
            locations=merged_mapping_df['iso_alpha3'],
            text=merged_mapping_df['country_name'],
            z=merged_mapping_df['total_bottle_weight'],
            colorscale=['green'] + ['yellow'] + ['orange'] * 5 + ['red'] * (len(merged_mapping_df) - 17),
            autocolorscale=False,
            marker_line_color='white',
            marker_line_width=0.5,
            colorbar=dict(
                title='Total Bottle<br>Weight (kg)',
                x=0,
                xanchor='left',
                len=0.9,
                y=0.9,
                yanchor='top',
                thickness=20,
                tickformat='.2f'
            ),
            zmin=0,
            zmax=np.percentile(merged_mapping_df['total_bottle_weight'], 95)
        ))

        fig.update_layout(
            title='Plastic Bottle Weight for Each Country',
            geo=dict(
                showframe=True,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            margin=dict(l=30, r=30, t=30, b=30)
        )

        for color, countries in colors.items():
            indices = [i for i, country in enumerate(merged_mapping_df['iso_alpha3']) if country in countries]
            fig.update_traces(marker=dict(color=color), selector=dict(indices=indices))

        fig.update_traces(hovertemplate='<b>%{text}</b><br>Bottle Weight: %{z:.4f}kg<extra></extra>')



        # Create the bar plot
        bar = go.Figure(data=go.Bar(
            x=total_bottle_weight_bar_df['total_bottle_weight'],
            y=total_bottle_weight_bar_df['country_name'],
            orientation='h'
        ))

        # Customize the layout
        bar.update_layout(
            title='Bottle Weight by Country',
            xaxis_title='Bottle Weight(kg)',
            yaxis_title='Country',
            bargap=0.1,
            bargroupgap=0.05,
            height=600,
            width=800
        )


        return html.Div([
            dcc.Graph(figure=fig),
            dcc.Graph(figure=bar)
        ])


# Callback to update the bar graph based on the selected metric
@app.callback(
    Output("bar-container", "children"),
    Input("dropdown-metric", "value")
)
def update_bar_graph(metric):
    if metric == 'bottles':
        # Create the horizontal bar plot for Bottles
        fig = create_bar_figures(metric)
        return dcc.Graph(figure=fig)

    elif metric == 'bottle_weight':
        # Create the horizontal bar plot for Bottle Weight
        fig = create_bar_figures(metric)
        return dcc.Graph(figure=fig)

    else:
        return html.Div("Invalid metric selected.")



def create_bar_figures(metric):
    if metric == 'bottles':
        # Create the horizontal bar plot for Bottles
        bar_bottles = go.Figure(data=go.Bar(
            x=top_10_manufacturers_count['bottle_count'],  # Bottle count
            y=top_10_manufacturers_count.index,  # Manufacturer names
            orientation='h'  # Horizontal orientation
        ))

        # Customize the layout of the plot
        bar_bottles.update_layout(
            title='Top 10 Manufacturers by Bottle Count',
            xaxis_title='Number of Bottles',
            yaxis_title='Manufacturer'
        )

        return bar_bottles

    elif metric == 'bottle_weight':
        # Create the horizontal bar plot for Bottle Weight
        bar_bottle_weight = go.Figure(data=go.Bar(
            x=top_10_manufacturers_weight['total_bottle_weight'],  # Total bottle weight
            y=top_10_manufacturers_weight.index,  # Manufacturer names
            orientation='h'  # Horizontal orientation
        ))

        # Customize the layout of the plot
        bar_bottle_weight.update_layout(
            title='Top 10 Manufacturers by Total Bottle Weight',
            xaxis_title='Total Bottle Weight(kgs)',
            yaxis_title='Manufacturer'
        )

        return bar_bottle_weight

    else:
        return None


# Define the callback function to update the manufacturer output based on the selected country
@app.callback(
    Output('manufacturer-output', 'children'),
    [Input('country-dropdown', 'value')]
)
def update_manufacturer_output(country_name):
    if country_name:
        # Filter the manufacturer_df based on the selected country name
        filtered_df = manufacturer_df[manufacturer_df['country_name'] == country_name].copy()

        # Calculate bottlecount_percent
        filtered_df['bottlecount_percent'] = (filtered_df['bottle_count'] / filtered_df['bottle_count'].sum()) * 100

        # Group the filtered_df by manufacturer_name and calculate the total bottlecount_percentage and sum of bottle_count for each manufacturer
        grouped_df = filtered_df.groupby('manufacturer_name').agg({'bottlecount_percent': 'sum', 'bottle_count': 'sum'})

        # Get the top 3 manufacturers and their corresponding total bottlecount_percentage and sum of bottle_count
        top_manufacturers = grouped_df.nlargest(3, 'bottle_count')

        # Create the manufacturer output text
        output_text = [html.H3("Top Manufacturers with the most Plastic bottles in the Environment:")]
        if len(top_manufacturers) > 0:
            for i, (manufacturer, row) in enumerate(top_manufacturers.iterrows(), start=1):
                percentage = row['bottlecount_percent']
                total_bottle_count = row['bottle_count']
                output_text.append(html.P("{}. Manufacturer: {}".format(i, manufacturer)))
                output_text.append(html.P("Total number of bottles in {} belonging to {}: {}".format(country_name, manufacturer, total_bottle_count)))
                output_text.append(html.P("That is {}% of all bottles in {}".format(percentage, country_name)))
                output_text.append(html.Br())
        else:
            output_text.append(html.P("No manufacturer names found for the specified country."))

        return output_text

    # Return an empty output if no country is selected
    return []


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
































