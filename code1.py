import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import tkinter as tk
from tkinter import ttk
import db
from tkinter import messagebox

# Load the dataset
data = pd.read_csv('project_dataset1.csv')
print(data)

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
    db.insert(part_name,brand ,condition,rating)
    
    if rating <= 5.0:
        predict_price(part_name, brand, condition, rating)
    else:
        result_label.config(text="Invalid rating")

# Create the main Tkinter window


# Create the main Tkinter window
root = tk.Tk()
root.title("Part Price Predictor")
root.geometry("400x400")  # Set the initial window size

# Create and pack a frame for input fields
input_frame = ttk.Frame(root)
input_frame.pack(pady=10)

# Create and pack widgets (labels, entry fields, buttons) with better styling
part_name_label = ttk.Label(input_frame, text="Enter the part name:")
part_name_label.grid(row=0, column=0, padx=10, pady=5)

part_name_entry = ttk.Entry(input_frame)
part_name_entry.grid(row=0, column=1, padx=10, pady=5)

brand_label = ttk.Label(input_frame, text="Enter the brand:")
brand_label.grid(row=1, column=0, padx=10, pady=5)

brand_entry = ttk.Entry(input_frame)
brand_entry.grid(row=1, column=1, padx=10, pady=5)

condition_label = ttk.Label(input_frame, text="Enter the condition:")
condition_label.grid(row=2, column=0, padx=10, pady=5)

condition_values = ["New", "Used", "Refurbished"]
condition_combobox = ttk.Combobox(input_frame, values=condition_values)
condition_combobox.grid(row=2, column=1, padx=10, pady=5)

rating_label = ttk.Label(input_frame, text="Enter the seller rating below 5.0:")
rating_label.grid(row=3, column=0, padx=10, pady=5)

rating_entry = ttk.Entry(input_frame)
rating_entry.grid(row=3, column=1, padx=10, pady=5)

predict_button = ttk.Button(input_frame, text="Predict", command=on_predict_button_click)
predict_button.grid(row=4, columnspan=2, pady=10)

# Create a result frame for displaying predictions
result_frame = ttk.Frame(root)
result_frame.pack(pady=10)

result_label = ttk.Label(result_frame, text="Predicted Sales for the Next 5 Years:")
result_label.pack()

predicted_sales_label = ttk.Label(result_frame, text="")
predicted_sales_label.pack()

# Function to show an error message
def show_error_message(message):
    messagebox.showerror("Error", message)

# Function to handle the button click event
def on_predict_button_click():
    part_name = part_name_entry.get().upper()
    brand = brand_entry.get().capitalize()
    condition = condition_combobox.get().capitalize()
    
    try:
        rating = float(rating_entry.get())
    except ValueError:
        show_error_message("Invalid rating. Please enter a numeric value.")
        return
    
    if 0 <= rating <= 5.0:
        db.insert(part_name, brand, condition, rating)
        predict_price(part_name, brand, condition, rating)
    else:
        show_error_message("Invalid rating. Please enter a value between 0 and 5.0.")

# Start the Tkinter main loop
root.mainloop()
