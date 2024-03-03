import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
data.keys()
print(data['target_names'])
print(data['feature_names'])
df1=pd.DataFrame(data['data'],columns=data['feature_names'])
scaling=StandardScaler()
scaling.fit(df1)
Scaled_data=scaling.transform(df1)
principal=PCA(n_components=3)
principal.fit(Scaled_data)
x=principal.transform(Scaled_data)
print(x.shape)
principal.components_
plt.figure(figsize=(10,10))
plt.scatter(x[:,0],x[:,1],c=data['target'],cmap='plasma')
plt.ylabel('pc1')
plt.ylabel('pc2')
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(10,10))
axis=fig.add_subplot(111,projection='3d')
axis.scatter(x[:,0],x[:,1],x[:,2],c=data['target'],cmap='plasma')
axis.set_xlabel("PC1",fontsize=10)
axis.set_ylabel("PC2",fontsize=10)
axis.set_zlabel("PC3",fontsize=10)
print(principal.explained_variance_ratio_)
