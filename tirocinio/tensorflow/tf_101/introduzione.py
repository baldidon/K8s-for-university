#esempio Tensorflow
#creare un modello in grado di riconoscere immagini vestiti del dataset MNIST
#in particolare il dataset fashion MNIST
#Questa guida usa il dataset Fashion MNIST che contiene 70,000 immagini in toni di grigio di 10 categorie. 
#Le immagini mostrano singoli articoli di abbigliamento a bassa risoluzione (28 per 28 pixel)

#librerie d'appoggio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#tensorflow libraries
import tensorflow as tf
from tensorflow import keras


#sto forzando ad usare tkinter per visualizzare i grafici di matplotlib 
mpl.use('TkAgg')
#selezioniamo il dataset, scaricandolo
fashion_mnist = keras.datasets.fashion_mnist

#carichiamo i dati (immagini) per allenare il mocdello e per testare quant'è accurato
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()
#le immagini possono essere interpretate come un array bidimensionale 28*28 (di numpy)in cui i valori dei pixel vanno da 0 a 255
#le etichette sono un vettore di interi, definiti da 0 a 9; ogni numero rappresenta la classe di immagine da riconoscere 
#in questo caso ogni classe è un tipo di capo d'abbigliamento 

#definisco una tupla con tutti i nomi delle classi(capi d'abbigliamento)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#verifichiamo che non abbia detto cavolate
#dovrebbe restituire il fatto che è formato da 60k oggetti, di dimensione 28*28
print(f"caratteristiche del dato train images: {train_images.shape}")

#verifichiamo che il vettore train_labels contiene 60k etichette, una per immagine
print(len(train_labels))

#verifichiamo forma di test_images
print(f"caratteristiche dell'oggetto test_images: {test_images.shape}")


#==================================================================
#prima di elaborare il modello dobbiamo preparare i dati
#plottiamo una immagine di uno dei due oggetti
#train_images = train_images / 255.0
#test_images = test_images / 255.0
#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

#pixel ha un valore di profondità da 0 a 250, scaliamo il tutto per ottenere un range di valori da 0 a 1 
train_images = train_images / 255.0
test_images = test_images / 255.0

#Per verificare che i dati siano nella forma corretta e che tutto sia pronto per costruire e allenare la rete,
#visualizziamo le prime 25 immagini del insieme di addestramento e visualizziamo il nome della classe sotto a ciascuna immagine.

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#==============================================================================================================

#L'elemento costruttivo di base di una rete neurale è il livello. 
#I livelli estraggono rappresentazioni dai dati con cui vengono alimentati. 
#Sperabilmente, queste rappresentazioni sono significative per il problema che si sta trattando.
#La maggior parte del deep learning consiste nel collegare tra loro livelli semplici. 
#La maggior parte dei livelli, come tf.keras.layers.Dense, hanno parametri che sono imparati durante l'allenamento

model = keras.Sequential(
    [
    #flatters serve per rendere un array bidimensionale in un array monodimensionale
    keras.layers.Flatten(input_shape=(28,28)),
    #questi due livelli sono molto importanti, sono livelli neurali
    # il primo mi genera 128 nodi
    keras.layers.Dense(128, activation='relu'),
    # il secondo invece mi genera 10 nodi, estituisce un vettore di 10 valori di probabilità la cui somma è 1. 
    # Ogni nodo contiene un valore che indica la probabilità che l'immagine corrente appartenga ad una delle 10 classi(0-9). 
    keras.layers.Dense(10, activation='softmax')
    ]
)

#per poter renedere il modello funzionante, devo tenere conto di 
#Funzione perdita —Misura quanto è accurato il modello durante l'apprendimento. La volontà è di minimizzare questa funzione per "dirigere" il modello nella giusta direzione.
#Ottimizzatore —Indica com'è aggiornato il modello sulla base dei dati che tratta e della sua funzione perdita.
#Metriche —Usate per monitorare i passi di addestramento e verifica. L'esempio seguente usa come accuratezza, la frazione delle immagini che sono classificate correttamente.
model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#adesso inizia la fase di apprendimento, in cui istruisco il modello con ciò che deve riconoscere e che deve fare
#Alimentare il modello con i dati di addestramento. In questo esempio, i dati di addestramento sono nei vettori train_images e train_labels
#così che sia in grado di fare le corrette associazioni
#Chiedere al modello di fare previsioni su un insieme di prova—in questo esempio, il vettore test_images.
#Verificare che le previsioni corrispondano alle etichette del vettore test_labels.
#Per iniziare l'addestramento, chiamare il metodo model.fit—chiamato così perchè "allena" il modello sui dati di addestramento

#epoch è il numero di "volte" che eseguo il test
# il test consiste nell'apprendimento di associare foto alla relativa etichetta
model.fit(train_images,train_labels,epochs=10)

#verifico l'accuratezza del mio modello
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#eseguendo lo script, otteniamo un'accuratezza di circa l'88 percento
