import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.spatial.distance import cdist

np.random.seed(0)

X_linear = np.linspace(0,10,100); y_linear = 2*X_linear+1+np.random.normal(0,2,100)
df_linear = pd.DataFrame({'X':X_linear,'Y':y_linear})

X_lwr = np.linspace(0,10,100); y_lwr = np.sin(X_lwr)+np.random.normal(0,0.1,100)
df_lwr = pd.DataFrame({'X':X_lwr,'Y':y_lwr})

X_poly = np.linspace(-3,3,100); y_poly = 0.5*X_poly**3 - X_poly**2 + X_poly + np.random.normal(0,1,100)
df_poly = pd.DataFrame({'X':X_poly,'Y':y_poly})

def linear_regression(df):
    X,y=df[['X']],df['Y']
    m=LinearRegression().fit(X,y)
    plt.scatter(X,y); plt.plot(X,m.predict(X),'r'); plt.title("Linear Regression"); plt.show()

linear_regression(df_linear)

def gaussian_kernel(x,X,tau): return np.exp(-cdist([[x]],X,'sqeuclidean')/(2*tau**2))

def locally_weighted_regression(X,y,tau=0.5):
    X=np.hstack([np.ones((X.shape[0],1)),X])
    xr=np.linspace(X[:,1].min(),X[:,1].max(),100); yp=[]
    for xi in xr:
        w=np.diag(gaussian_kernel(xi,X[:,1:],tau).flatten())
        t=np.linalg.pinv(X.T@w@X)@(X.T@w@y)
        yp.append([1,xi]@t)
    plt.scatter(X[:,1],y); plt.plot(xr,yp,'r'); plt.title("Locally Weighted Regression"); plt.show()

locally_weighted_regression(df_lwr[['X']].values,df_lwr['Y'].values)

def polynomial_regression(df,d=3):
    X,y=df[['X']],df['Y']
    m=make_pipeline(PolynomialFeatures(d),LinearRegression()).fit(X,y)
    plt.scatter(X,y); plt.plot(X,m.predict(X),'r'); plt.title(f"Polynomial Regression (deg={d})"); plt.show()

polynomial_regression(df_poly,3)
