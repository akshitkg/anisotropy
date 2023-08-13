import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import esda
from splot.esda import moran_scatterplot

# Step 1: Load the data and extract latitude and longitude
# Replace 'your_file.xlsx' with the actual path to your Excel file
file_path = 'data.xlsx'
data = pd.read_excel(file_path)

# Assuming your latitude and longitude columns are named 'Latitude' and 'Longitude'
latitude = data['Lat']
longitude = data['Long']

# Step 2: Convert latitude and longitude to pixel coordinates
# Choose an appropriate scale factor to convert degrees to pixels, adjust it based on your data
scale_factor = 100
x = (longitude - longitude.min()) * scale_factor
y = (latitude - latitude.min()) * scale_factor

# Step 3: Create a Moran object for spatial autocorrelation
attribute_values = data['Elevation']  # Replace 'Elevation' with your attribute column name <-- Unsure about this
w = esda.weights.Queen.from_dataframe(data)
moran = esda.Moran(attribute_values, w)

# Step 4: Visualize spatial autocorrelation using Moran's scatterplot
moran_scatterplot(moran)
plt.show()

print("Moran's I:", moran.I)
