import tkinter as tk
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Load the data
data = pd.read_csv('transfusion.data')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train a decision tree classifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X, y)

# rename the columns
data = data.rename(columns={'Recency (months)': 'Recency',
                            'Frequency (times)': 'Frequency',
                            'Monetary (c.c. blood)': 'Monetary',
                            'Time (months)': 'Time',
                            'whether he/she donated blood in March 2007': 'Donated_Mar_2007'})

# Create the Tkinter window
window = tk.Tk()
window.title('Blood Donation Predictor')
window.geometry('800x800')
window.configure(bg="white")

# Define the input fields
recency_label = tk.Label(window, text='Enter the number of months since the last donation:')
recency_label.pack()
recency_entry = tk.Entry(window)
recency_entry.pack()

frequency_label = tk.Label(window, text='Enter the number of times the donor has donated blood:')
frequency_label.pack()
frequency_entry = tk.Entry(window)
frequency_entry.pack()

monetary_label = tk.Label(window, text='Enter the total amount of blood donated in cubic centimeters:')
monetary_label.pack()
monetary_entry = tk.Entry(window)
monetary_entry.pack()

time_label = tk.Label(window, text='Enter the number of months since the donors first donation')
time_label.pack()
time_entry = tk.Entry(window)
time_entry.pack()

donated_label = tk.Label(window, text='Enter 1 if the donor has donated blood in March 2007, 0 if not')
donated_label.pack()
donated_entry = tk.Entry(window)
donated_entry.pack()

# Define the modified predict function
def predict():
    try:
        # Get the user inputs
        recency = recency_entry.get()
        frequency = frequency_entry.get()
        monetary = monetary_entry.get()
        time = time_entry.get()

        # Check if inputs are valid
        if not all(x.isnumeric() for x in [recency, frequency, monetary, time]):
            error_message.pack()
            return
        if float(recency) < 0 or float(frequency) < 0 or float(monetary) < 0 or float(time) < 0:
            error_message.pack()
            return

        # Make the prediction
        prediction = tree.predict([[recency, frequency, monetary, time]])

        # Show the prediction
        if prediction == 1:
            prediction_label.config(text='The person is likely to donate blood again.')
        else:
            prediction_label.config(text='The person is unlikely to donate blood again.')
    except:
        error_message.pack()

# Define the predict button
predict_button = tk.Button(window, text='Predict', command=predict, bg='red', fg='white', font=('helvetica', 9, 'bold'))
predict_button.pack(pady=10)

# Define the visualizations function and display in window
def show_visualizations():
    # Calculate the mean and standard deviation of the amount donated by each person in cc
    donation_stats = data.groupby('Donated_Mar_2007')['Monetary'].agg(['mean', 'std']).reset_index()

    # Create the bar chart
    fig2 = go.Figure(data=[go.Bar(x=donation_stats['Donated_Mar_2007'], y=donation_stats['mean'],
                                  error_y=dict(type='data', array=donation_stats['std']))])
    fig2.update_layout(title='Mean Amount Donated by Each Person (cc)',
                       xaxis_title='Donated',
                       yaxis_title='Mean Amount Donated (cc)')
    fig2_data = fig2.to_html(full_html=False, default_height=500, default_width=700)
    fig2_div = tk.Label(window, text=fig2_data, bg='white')
    fig2_div.pack(pady=10)

    # Create the scatter chart of total number of donations
    fig3 = px.scatter(data, x='Monetary', y='Frequency', color='Donated_Mar_2007')
    fig3.update_layout(title='Total Number of Donations vs. Frequency of Donations',
                       xaxis_title='Total Number of Donations (cc)',
                       yaxis_title='Frequency of Donations (times)')
    fig3_data = fig3.to_html(full_html=False, default_height=500, default_width=700)
    fig3_div = tk.Label(window, text=fig3_data, bg='white')
    fig3_div.pack(pady=10)

    # Create the chart of the highest donation in cc for blood
    fig4 = px.histogram(data, x='Monetary', color='Donated_Mar_2007', nbins=25, range_x=(0, 1500))
    fig4.update_layout(title='Highest Donation in cc for Blood',
                       xaxis_title='Amount Donated (cc)',
                       yaxis_title='Count')
    fig4.update_traces(opacity=0.75)
    fig4_data = fig4.to_html(full_html=False, default_height=500, default_width=700)
    fig4_div = tk.Label(window, text=fig4_data, bg='white')
    fig4_div.pack(pady=10)

    # Create the chart of the standard deviation of recency (months)
    fig5 = px.histogram(data, x='Recency', color='Donated_Mar_2007', nbins=25, range_x=(0, 50))
    fig5.update_layout(title='Standard Deviation of Recency (months)',
                       xaxis_title='Recency (months)',
                       yaxis_title='Count')
    fig5.update_traces(opacity=0.75)
    fig5_data = fig5.to_html(full_html=False, default_height=500, default_width=700)
    fig5_div = tk.Label(window, text=fig5_data, bg='white')
    fig5_div.pack(pady=10)

# Define the visualizations button
visualizations_button = tk.Button(window, text='Show Visualizations', command=show_visualizations, bg='red', fg='white', font=('helvetica', 9, 'bold'))
visualizations_button.pack(pady=10)

# Define the prediction label
prediction_label = tk.Label(window, text='Likelihood of Donating Blood: ', bg='white')
prediction_label.pack(pady=10)

# Define the error message
error_message = tk.Label(window, text='Please enter valid inputs (numerical values only)', fg='red', bg='white', font=('helvetica', 10, 'bold'))
instructions_label = tk.Label(window, text='Example input: 2, 50, 12500, 98, 1', bg='white')
instructions_label.pack()

# Run the Tkinter event loop
window.mainloop()