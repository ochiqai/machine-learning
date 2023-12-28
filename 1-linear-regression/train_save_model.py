import pickle
import numpy as np
from sklearn.linear_model import LinearRegression



## data
xonalar = [1, 2, 3, 5, 6, 7]
narxlar = [150, 200, 250, 350, 400, 450]

# listni numpyga o'tkazishimiz kerak, chunki sklearn numpy asosida ishlaydi
X = np.array(xonalar).reshape(-1, 1)
y = np.array(narxlar)
print(X.shape)
print(y.shape)

#train qilish jarayoni
model = LinearRegression()
model.fit(X, y)
with open('models/linearReg-Samariddin.pkl','wb') as f:
    pickle.dump(model,f)