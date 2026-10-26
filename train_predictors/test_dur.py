from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Replace these with your arrays or columns
data = np.loadtxt("/cfs/home/u035986/spkanon/spkanon_eval/private_knnvc/logs/train_dur_predictor/1775418533/test_predictions.txt")

y_true = data[:, 1] 
y_pred = data[:, 2] 

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_true, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

