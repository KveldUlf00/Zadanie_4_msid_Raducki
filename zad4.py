import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Importuje, a następnie ładuje dane Fashion MNIST bezpośrednio z TensorFlow
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Etykiety są tablicą liczb całkowitych z zakresu od 0 do 9. Odpowiadają one klasie odzieży, którą reprezentuje obraz.
# Każdy obraz jest przypisany do jednej etykiety. Ponieważ nazwy klas nie są zawarte w zestawie danych, należy je
# przechowywać tutaj, aby użyć ich później podczas drukowania obrazów:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Przeglądam dane
print(f'train_images.shape; {train_images.shape}')  # w zestawie uczącym jest 60000 obrazów o wymiarach 28x28 pikseli
print(f'len(train_labels): {len(train_labels)}')    # tak samo w zestawie uczącym jest 60000 etykiet
print(f'train_labels: {train_labels}')              # Każda etykieta jest liczbą całkowitą od 0 do 9
print(f'test_images.shape: {test_images.shape}')    # w zestawie testowym jest 10000 obrazów o wymiarach 28x28 pikseli
print(f'len(test_labels): {len(test_labels)}')      # tak samo w zestawie testowym jest 10000 etykiet

#                                   Wstępne przetworzenie danych
# Skaluje te wartości pikseli do zakresu od 0 do 1
train_images = train_images / 255.0
test_images = test_images / 255.0
# tutaj moge użyć preprocesora jak w zadaniu 3


#                                   Budowanie modelu
# Budowa wymaga skonfigurowania warstw modelu, a następnie skompilowania modelu

### Konfigurowaneie warstwy
# Warstwy wyodrębniają reprezentacje z wprowadzonych do nich danych
# Większość warstw, takich jak tf.keras.layers.Dense, ma parametry, których uczy się podczas treningu
# Pierwsza warstwa Flatten przekształca format obrazów z dwuwymiarowej tablicy (28 na 28 pikseli) do jednowymiarowej
#   tablicy (28 * 28 = 784 piksele). Ta warstwa nie ma parametrów do nauczenia, tylko formatuje dane
# Po spłaszczeniu pikseli sieć składa się z sekwencji dwóch warstw tf.keras.layers.Dense
# Pierwsza warstwa Dense ma 128 węzłów (lub neuronów).
# Druga warstwa zwraca tablicę logitów o długości 10. Każdy węzeł zawiera punktację wskazującą, że bieżący obraz należy
#   do jednej z 10 klas.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

### Kompilacja modelu
# Zanim model będzie gotowy do treningu, potrzebuje kilku ustawień. Są one dodawane podczas etapu kompilacji modelu:
# - Funkcja straty - mierzy dokładność modelu podczas treningu. Minimalizujemy tę funkcję, aby „sterować” modelem
#       we właściwym kierunku
# - Optymalizator - w ten sposób model jest aktualizowany na podstawie widzianych danych i funkcji utraty
# - Metryki - używane do monitorowania etapów szkolenia i testowania. Poniższy przykład używa dokładności,
#       czyli ułamka obrazów, które są poprawnie sklasyfikowane.
model.compile(optimizer='Adagrad',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments


#                                       Trenowanie modelu
### Karmienie modelu
# Aby rozpocząć trenowanie, wywołuje metodę model.fit - tak zwaną, ponieważ „dopasowuje” ona model do danych model.fit
model.fit(train_images, train_labels, epochs=600)
# Gdy model uczy się, wyświetlane są metryki strat i dokładności

### Ocenianie dokładności
# Porównuje jak model działa na testowym zestawie danych
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print('Test loss:', test_loss)
# Okazuje się, że dokładność zestawu danych testowych jest nieco mniejsza niż dokładność zestawu danych uczących.
# Ta luka między dokładnością treningu a dokładnością testu oznacza nadmierne dopasowanie. Do nadmiernego dopasowania
# dochodzi, gdy model uczenia maszynowego działa gorzej na nowych, wcześniej niewidocznych danych wejściowych,
# niż w przypadku danych szkoleniowych. Przekrojony model „zapamiętuje” szum i szczegóły w zbiorze danych uczących do
# punktu, w którym ma to negatywny wpływ na wydajność modelu w nowych danych

### Prognozowanie
print('\n==================== Prognozowanie ===================')
# Po przeszkoleniu modelu można go używać do prognozowania niektórych obrazów
# Dołączam warstwę softmax, konwertującą logity na prawdopodobieństwa, które są łatwiejsze do zinterpretowania
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)    # Tutaj model przewidział etykietę dla każdego obrazu w zestawie testowym
print(f'Lista prognóz: {predictions[0]}')              # Pierwsza prognoza
print(f'Największa wartość z listy: {np.argmax(predictions[0])}')    # która z prognoz ma największą wartość
# Tak więc model jest najbardziej pewny, że ten obraz to but za kostkę lub class_names[9]
print(f'Rzeczywista wartość etykiety: {test_labels[0]}')  # Analiza etykiety testowej pokazuje, że ta klasyfikacja jest poprawna

print('\nPredykcja === Rzeczywistość')
for i in range(40):
    if np.argmax(predictions[i]) == test_labels[i]:
        wynik = True
    else:
        wynik = False
    print(f'        {np.argmax(predictions[i])} === {test_labels[i]} ||| {wynik}')

### Używanie wytrenowanego modelu
print('\n==================== Używanie wytrenowanego modelu ============')
# Używam wytrenowanego modelu, aby przewidzieć pojedynczy obraz
img = test_images[3]
# print(f'img.shape: {img.shape}')  # (28,28)

# Modele tf.keras są zoptymalizowane pod kątem tworzenia prognozy na zbiorze wielu przykładów w tym samym momencie
# W związku z tym, nawet jeśli używam pojedynczego obrazu, musze go dodać do listy
img = (np.expand_dims(img,0))
# print(f'img.shape: {img.shape}')  # (1, 28, 28)

# Typuje poprawną etykietę dla tego obrazu:
predictions_single = probability_model.predict(img)
print(f'Predictions_single: {predictions_single}')

# tf.keras.Model.predict zwraca listę list - jedną listę dla każdego obrazu w tf.keras.Model.predict danych.
# Pobieram prognozy dla naszego (jedynego) obrazu w partii
print(f'Prognoza etykiety obrazu: {np.argmax(predictions_single[0])}')

# To o to chodziło
print(f'Rzeczywista etykieta obrazu: {test_labels[3]}')
plt.figure()
plt.imshow(test_images[3])
plt.colorbar()
plt.grid(False)
plt.show()
