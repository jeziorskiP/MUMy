import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score
from sklearn.metrics  import silhouette_samples
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

from scipy.cluster.hierarchy import dendrogram





"""
customer_data = pd.read_csv('D:\Programy\Class\glass.csv')
data = customer_data.iloc[:, 0:10].values      #glass 7
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
datasetname = 'glass/'
"""
"""
customer_data = pd.read_csv('D:\Programy\Class\Cancer.csv')
data = customer_data.iloc[:, 0:9].values      #cancer 2
range_n_clusters = [2, 3, 4 ,5]
datasetname = 'cancer/'
"""
"""
customer_data = pd.read_csv('D:\Programy\Class\Banknote.csv')
data = customer_data.iloc[:, 0:4].values      #banknote 2
range_n_clusters = [2, 3, 4, 5]
datasetname = 'banknote/'
"""

customer_data = pd.read_csv('D:\Programy\Class\zb1.csv')
data = customer_data.iloc[:, 0:2].values      
range_n_clusters = [2, 3, 4,5,6,7,8]
datasetname = 'zb1/'



print(data)

#cluster = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
#cluster.fit_predict(data)


#range_n_clusters = [2, 3, 4, 5, 6,7]
#datasetname = 'glass/'

#“euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”.
#range_affinity = [ "manhattan"]
#range_affinity = [ "euclidean"]
range_affinity = ['euclidean', "manhattan"]

#{“ward”, “complete”, “average”, “single”},
#range_linkage = ["complete", "average", "single"]
range_linkage = [ "ward","complete", "average", "single"]
#range_linkage = [ "ward"]

#range_n_clusters = [2, 3, 4, 5, 6,7]
katalog = "./schemat/"
msg =""

for affinity in range_affinity:
    for linkage in range_linkage:
        katalog = "./schemat/"
        msg =""
        katalog = katalog + affinity+linkage+"/" + datasetname
        print(katalog)

        for n_clusters in range_n_clusters: 
            if affinity == 'manhattan' and linkage == 'ward':
                break
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                #ax1.plt.color = rainbow
                cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
                preds = cluster.fit_predict(data)

                score = silhouette_score(data, preds)
                sample_silhouette_values = silhouette_samples(data, preds)


                print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
                calc = n_clusters/score
                msg = msg + str((n_clusters)) +" " +  str(round(score,3)) + " " +str(round(calc, 3))+ "\n "
                txt=""
                txt = txt +katalog + "wyniki.txt"
                print(msg,  file=open(txt, 'w'))
                
                print("DS")
                print(sample_silhouette_values)
                
                
                plt.figure(figsize=(10, 7))
                ccc = cm.nipy_spectral(preds.astype(float) / n_clusters)
                plt.title("Data n_clusters {}".format(n_clusters) )
                plt.scatter(data[:,0], data[:,1], c=ccc, cmap='rainbow')
                name = ""
                name = name + katalog + "wykresN" +str(n_clusters)
                plt.savefig(name + ".jpg")

                
                y_lower = 10 
                
                for i in range(n_clusters):
                    print("Jestesm w petli {}".format(i))
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = \
                        sample_silhouette_values[preds == i]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = cm.nipy_spectral(float(i) / n_clusters)
                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                    # Label the silhouette plots with their cluster numbers at the middle
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples



                print("Jestesm POZA")
                ax1.set_title("The silhouette plot")
                ax1.set_xlabel("The silhouette coefficient values")
                ax1.set_ylabel("Cluster label")
                
                # The vertical line for average silhouette score of all the values
                ax1.axvline(x=score, color="red", linestyle="--")

                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
                plt.savefig("photo.jpg")
                name = ""
                name = name + katalog + "qualityN" +str(n_clusters)
                fig.savefig(name + ".jpg")
                
                colors = cm.nipy_spectral(preds.astype(float) / n_clusters)
                ax2.scatter(data[:, 0], data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
                """
                ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')
                for i, c in enumerate(centers):
                    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')
"""
                ax2.set_title("Clustered data.")
                ax2.set_xlabel("Feature space for the 1st feature")
                ax2.set_ylabel("Feature space for the 2nd feature")
                fig.suptitle(("The visualization of the clustered data"
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
                
                
                fig.savefig(name + "1.jpg")
                
                plt.close('all')
        plt.show()
    

print("--------------------------------------------------------------")
"""
plt.figure(figsize=(10, 7))
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')


plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))
"""
