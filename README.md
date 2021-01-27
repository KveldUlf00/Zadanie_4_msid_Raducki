**Błażej Raducki 254277**


### Introduction: ###  

  W zadaniu 3 sprawdzam jakość modelu regresji logistycznej na zbiorze FashionMnist. W tym celu wyuczyłem model na danych trenigowych i 
	oceniłem jego jakość na danych testowych w oparciu o poprawność klasyfikacji. Natomiast w zadaniu 4 wybrałem skorzystałem z sieci neuronowej
	do klasyfikacji zbioru FashionMnist.

### Methods: ### 

W tych zadaniach używałem bibliotek tensorflow, sklearn oraz numpy, pandas i matplotlib. Najważniejszymi metodami używanymi z tych
bibliotek są read_csv() używane do pobrana danych, preprocessing.MinMaxScaler() i minmax_scale.fit_transform() do skalowania danych, 
LogisticRegression() jako model regresji logistycznej, predict() używaną w procesie sprawdzania uczenia, keras.Sequential() pomocna do 
konfiguracji warstw modelu, compile() używana w procesie kompilacji modelu. Przy korzystaniu z funkcji używałem takich stron jak :
  
* https://www.tensorflow.org/  
* https://byes.pl/wp-content/uploads/danologia/W13_koderskie_podsumowanie.pdf
* https://scikit-learn.org/stable/index.html


### Results: ###  

W zadaniu 3 używałem modelu regresji logistycznej używając parametrów: penalty: l2, odwrotność siły regularyzacji C: 1/10/100,
multi_class: ovr, oraz max_iter: 2000-4000 (zmieniana wartość iteracji branych dla rozwiązania konwergencji), wyniki:
  
C | Max_iter | Training accuracy | Validation accuracy | Benchmark
--|----------|-------------------|---------------------|----------
1 | 2000 | 0.87386 | 0.8515 | 0.842
10 | 3000 | 0.87804 | 0.8497 | 0.839
100 | 4000 | 0.88008 | 0.8473 | 0.838

W zadaniu 4 zastosowałem sieć neuronową z parametrami: optimizer='Adam/Adagrad', 
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'], oraz zmieniając liczbe epok, tak więc dla 
  
**optimizer='Adam'**
epochs | Test accuracy | Test loss
-------|---------------|----------
100 | 0.8863000273704529 | 0.8620233535766602
200 | 0.8805000185966492 | 1.2295786142349243
400 | 0.8780999779701233 | 2.0731465816497803
600 | 0.887499988079071  | 2.4908032417297363
800 | 0.8815000057220459 | 2.8832852840423584

**optimizer='Adagrad'**
epochs | Test accuracy | Test loss
-------|---------------|----------
100 | 0.8507000207901001 | 0.424144983291626
200 | 0.8589000105857849 | 0.40255504846572876
400 | 0.8702999949455261 | 0.3750693202018738 
600 | 0.8700000047683716 | 0.3686307668685913
800 | 0.8758000135421753 | 0.3570008873939514
  

### Usage: ###  

Aby korzystać z zadań, należy mieć dostęp do dowolnego środowiska programistycznego języka Python z wgranymi bibliotekami tensorflow, 
sklearn, numpy, pandas i matplotlib. Programy uruchamia się w zwyczajny sposób RUN. Dodatkowo aby włączyć zadanie 3 potrzebne są 
pliki csv z danymi (udostępnione w dysku google, link poniżej), w zadaniu 4 dane pobierane są automatyczne.

Link do dysku z danymi: https://drive.google.com/drive/folders/1IPwK4ZBEjqCjmvHFoX5dXgYx9j7cISPr?usp=sharing
