import joblib 
import numpy as np
loaded_model = joblib.load('best_knn_model.pkl')
new_data_point = np.array([[19, 10, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]])
prediction = loaded_model.predict(new_data_point)
print(loaded_model.weights)
print("Tahmin:", prediction)

print("dir(loaded_model")
print(dir(loaded_model))
