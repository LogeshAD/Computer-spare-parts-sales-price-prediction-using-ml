import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import tkinter as tk
from tkinter import ttk

# Load the dataset
data = pd.read_csv('project_dataset1.csv')

# Function to preprocess user input and make predictions
def predict_price(part_name, brand, condition, rating):
    # Filter the dataset for the user-entered part, brand, and condition
    filtered_data = data[(data['part_type'] == part_name) & (data['brand'] == brand) & (data['condition'] == condition)]

    if filtered_data.empty:
        result_label.config(text=f"No data found for {brand} {part_name} in {condition} condition.")
        return

    # Group the filtered data by year and calculate the total sales for each year
    sales_by_year = filtered_data.groupby('year')['sale_price'].sum().reset_index()

    # Create a feature matrix X (years) and target variable y (sales)
    X = sales_by_year['year'].values.reshape(-1, 1)
    y = sales_by_year['sale_price'].values

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the sales for the next 5 years with random variations
    future_years = [max(sales_by_year['year']) + i for i in range(1, 6)]
    future_sales = []

    for _ in range(5):
        # Predict the price without random variation
        predicted_price = model.predict(np.array([future_years[_]]).reshape(-1, 1))[0]
        
        # Add some random variation (you can customize this part)
        variation = random.uniform(-500, 500)  # Example: Allow prices to vary by up to $500
        predicted_price_with_variation = predicted_price + variation
        
        future_sales.append(predicted_price_with_variation)

    # Calculate accuracy metrics
    mae = mean_absolute_error(y, model.predict(X))
    mse = mean_squared_error(y, model.predict(X))
    rmse = np.sqrt(mse)

    # Create a bar chart to visualize the predicted sales for the next 5 years
    plt.figure(figsize=(8, 6))
    plt.bar(future_years, future_sales, color='red', label='Predicted Sales')
    plt.xlabel('Year')
    plt.ylabel('Predicted Sales Price')
    plt.title(f'Predicted Sales for {brand} {part_name} ({condition} Condition) for Next 5 Years\n')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Instead of plt.show(), save the plot to a file (optional)
    plt.savefig('predicted_sales_plot.png')
    plt.show()
    
    # Update a label or text widget in the GUI to display the image or a message
    result_label.config(text="Predicted Sales for the Next 5 Years:")
    
    # Create a string to display predicted sales
    predicted_sales_text = ""
    for year, sales in zip(range(max(data['year']) + 1, max(data['year']) + 6), future_sales):
        predicted_sales_text += f"Year {year}: ${sales:.2f}\n"
    
    predicted_sales_label.config(text=predicted_sales_text)

# Function to handle the button click event
def on_predict_button_click():
    part_name = part_name_entry.get().upper()
    brand = brand_entry.get().capitalize()
    condition = condition_combobox.get().capitalize()
    rating = float(rating_entry.get())
    
    if rating <= 5.0:
        predict_price(part_name, brand, condition, rating)
    else:
        result_label.config(text="Invalid rating")

# Create the main Tkinter window
root = tk.Tk()
root.title("Part Price Predictor")

# Create and pack widgets (labels, entry fields, buttons)
part_name_label = tk.Label(root, text="Enter the part name:")
part_name_label.pack()

part_name_entry = tk.Entry(root)
part_name_entry.pack()

brand_label = tk.Label(root, text="Enter the brand:")
brand_label.pack()

brand_entry = tk.Entry(root)
brand_entry.pack()

condition_label = tk.Label(root, text="Enter the condition (new/used/refurbished):")
condition_label.pack()

condition_values = ["New", "Used", "Refurbished"]
condition_combobox = ttk.Combobox(root, values=condition_values)
condition_combobox.pack()

rating_label = tk.Label(root, text="Enter the seller rating below 5.0:")
rating_label.pack()

rating_entry = tk.Entry(root)
rating_entry.pack()

predict_button = tk.Button(root, text="Predict", command=on_predict_button_click)
predict_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

predicted_sales_label = tk.Label(root, text="")
predicted_sales_label.pack()

# Start the Tkinter main loop
root.mainloop()
