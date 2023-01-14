# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 10:01:46 2023

@author: mehmet esen
"""
#from collections import Counter: Bu satır, 
#Python'da sayım işlemleri için kullanılan
# "Counter" nesnesini içerir. 
#Bu nesne, verilen bir liste veya 
#sözlük içindeki elemanların sayısını saklar.
from collections import Counter
#from sklearn.datasets import load_iris:
#Bu satır, scikit-learn kütüphanesinden Iris veri kümesini yükler. 
#Bu veri kümesi, üç farklı çiçek türü olan iris çiçekleri
#için özellikleri içerir.
from sklearn.datasets import load_iris
#from sklearn.metrics import accuracy_score :
#Bu satır scikit-learn kütüphanesinden accuracy_score metodunu import eder.
#Bu metod yapılan tahminlerin doğruluk oranını hesaplamak için kullanılır.
from sklearn.metrics import accuracy_score

#Bu metod veri setini eğitim ve test verilerine ayırmak için kullanılır.
from sklearn.model_selection import train_test_split

# Veri kümesini yükle
iris = load_iris()

# Veri kümesini eğitim ve test verilerine ayır
#Verilerin yüzde 20'si test verileri olarak kullanılır.
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
#Bu fonksiyon, eğitim verileri, eğitim verilerinin etiketleri,
# test verisi ve k değişkenini alır.
def knn_algoritma(X_train, y_train, x_test, k):
    distances = []
    for i in range(len(X_train)):
        # bu satır öklit mesafesini hesaplar
        #eğitim verisi ile test verisi arasındaki mesafeyi hesaplar. 
        distance = ((X_train[i,:] - x_test)**2).sum()
        #hesaplanan mesafe ile eğitim verisi
        #arasındaki indeksi distances listesine ekler.
        distances.append([distance, i])
        #mesafeler listesini sıralar. Bu işlem,
        #en yakın komşuların bulunmasını kolaylaştırır.
    distances = sorted(distances)
    # en yakın komşuların etiketlerini saklamak için bir liste oluşturur.
    targets = []
    #Bu döngü, en yakın k komşunun etiketlerini bulmak için kullanılır.
    for i in range(k):
        #Bu satır, en yakın komşunun eğitim verisi arasındaki indeksini alır.
        index_of_training_data = distances[i][1]
        #en yakın komşunun etiketini targets listesine ekler.
        targets.append(y_train[index_of_training_data])
        #en çok tekrar eden etiketi döndürür. Bu etiket, 
        #test verisi için tahmin edilen etikettir.
    return Counter(targets).most_common(1)[0][0]

# Test verilerini sınıflandır
predictions = []
k = 5 #en yakın komşuların sayısını belirler.
for x_test in X_test: #test verileri için tahmin yapmak için kullanılır.
    #test verisi için yapılan tahmini predictions listesine ekler.
    predictions.append(knn_algoritma(X_train, y_train, x_test, k))

# Doğruluk oranını hesapla
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", accuracy)
#her bir iris çiçeği için "setosa", 
#"versicolor" veya "virginica" gibi bir etiket verilmiştir.




#Iris veri kümesinde, her bir iris çiçeği için
#dört adet bağımsız değişken kullanılmıştır.
#Bu değişkenler, iris çiçeklerinin fiziksel özelliklerini
#ifade eder ve şöyle sıralanabilir:

    """
 SepalLengthCm: Iris çiçeğinin sepals (yapraklar) uzunluğu, 
 santimetre cinsinden ölçülmüştür.

SepalWidthCm: Iris çiçeğinin sepals (yapraklar) genişliği,
 santimetre cinsinden ölçülmüştür.

PetalLengthCm: Iris çiçeğinin petals (çiçek yaprakları) uzunluğu, 
santimetre cinsinden ölçülmüştür.
 PetalWidthCm : Iris çiçeğinin petals (çiçek yaprakları) genişliği, 
 santimetre cinsinden ölçülmüştür.
    """















