import tkinter as tk
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

# Load the data
data = pd.read_csv('transfusion.data')

# rename the columns
data = data.rename(columns={'Recency (months)': 'Recency',
                            'Frequency (times)': 'Frequency',
                            'Monetary (c.c. blood)': 'Monetary',
                            'Time (months)': 'Time',
                            'whether he/she donated blood in March 2007': 'Donated_Mar_2007'})

# Create the Tkinter window
window = tk.Tk()
window.title('Blood Donation Predictor')
window.geometry('600x600')

# Define the input fields
recency_label = tk.Label(window, text='Recency (Months)')
recency_label.pack()
recency_entry = tk.Entry(window)
recency_entry.pack()

frequency_label = tk.Label(window, text='Frequency of Donations')
frequency_label.pack()
frequency_entry = tk.Entry(window)
frequency_entry.pack()

monetary_label = tk.Label(window, text='Monetary')
monetary_label.pack()
monetary_entry = tk.Entry(window)
monetary_entry.pack()

time_label = tk.Label(window, text='Time')
time_label.pack()
time_entry = tk.Entry(window)
time_entry.pack()

donated_label = tk.Label(window, text='whether he/she donated blood in March 2007')
donated_label.pack()
donated_entry = tk.Entry(window)
donated_entry.pack()

# Define the error message
error_message = tk.Label(window, text='Please enter valid inputs', fg='red')

# Define the predict function
def predict():
    # Get the user inputs
    recency = recency_entry.get()
    frequency = frequency_entry.get()
    monetary = monetary_entry.get()
    time = time_entry.get()
    donated = donated_entry.get()

    # Check if inputs are valid
    if not recency.isdigit() or not frequency.isdigit() or not monetary.isdigit() or not time.isdigit() or not donated.isdigit():
        error_message.pack()
        return

    # Make the prediction
    prediction = model.predict(np.array([[int(recency), int(frequency), int(monetary), int(time), int(donated)]]))[0]

    # Show the prediction
    prediction_label.config(text='Likelihood of Donating Blood: {}%'.format(round(prediction * 100, 2)))

# Define the predict button
predict_button = tk.Button(window, text='Predict', command=predict)
predict_button.pack()

# Define the visualizations button
def show_visualizations():
    # Calculate the mean and standard deviation of the amount donated by each person in cc
    donation_stats = data.groupby('Donated_Mar_2007')['Monetary'].agg(['mean', 'std']).reset_index()

    # Create the bar chart
    fig2 = go.Figure(data=[go.Bar(x=donation_stats['Donated_Mar_2007'], y=donation_stats['mean'],
                                  error_y=dict(type='data', array=donation_stats['std']))])
    fig2.update_layout(title='Mean Amount Donated by Each Person (cc)',
                       xaxis_title='Donated',
                       yaxis_title='Mean Amount Donated (cc)')
    fig2.show(renderer='browser', auto_open=False)

    # Create the scatter chart of total number of donations
    fig3 = px.scatter(data, x='Monetary', y='Frequency', color='Donated_Mar_2007')
    fig3.update_layout(title='Total Number of Donations vs. Frequency of Donations',
                       xaxis_title='Total Number of Donations (cc)',
                       yaxis_title='Frequency of Donations (times)')
    fig3.show(renderer='browser', auto_open=False)

    # Create the chart of the highest donation in cc for blood
    fig4 = px.histogram(data, x='Monetary', color='Donated_Mar_2007', nbins=25, range_x=(0, 1500))
    fig4.update_layout(title='Highest Donation in cc for Blood',
                       xaxis_title='Amount Donated (cc)',
                       yaxis_title='Count')
    fig4.update_traces(opacity=0.75)
    fig4.show(renderer='browser', auto_open=False)

    # Create the chart of the standard deviation of recency (months)
    fig5 = px.histogram(data, x='Recency', color='Donated_Mar_2007', nbins=25, range_x=(0, 50))
    fig5.update_layout(title='Standard Deviation of Recency (months)',
                       xaxis_title='Recency (months)',
                       yaxis_title='Count')
    fig5.update_traces(opacity=0.75)
    fig5.show(renderer='browser', auto_open=False)

# Define the visualizations button
visualizations_button = tk.Button(window, text='Show Visualizations', command=show_visualizations)
visualizations_button.pack()

# Define the prediction label
prediction_label = tk.Label(window, text='Likelihood of Donating Blood: ')
prediction_label.pack()

# Run the Tkinter event loop
window.mainloop()