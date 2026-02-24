from ultralytics.utils.plotting import plot_results

# Point this directly to the results.csv file (or the folder containing it)
plot_results('runs/Anti-UAV/yolo11n-RGBRGB6C-midfusion/results.csv')

print("Success! Check your folder for the new results.png image.")