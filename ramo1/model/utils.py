import csv # Importa la libreria 'csv' per gestire i file in formato CSV.
import random # Importa la libreria 'random' per generare numeri casuali.
from functools import partialmethod # Importa la classe 'partialmethod' dal modulo 'functools' per creare metodi parziali.

import torch # Importa la libreria PyTorch.
import numpy as np # Importa la libreria NumPy, spesso usata per lavorare con array.
from sklearn.metrics import precision_recall_fscore_support # Importa una funzione da scikit-learn per calcolare metriche di valutazione.

class AverageMeter(object): # Definizione della classe AverageMeter. L'ereditarietà da 'object' è implicita in Python 3.
    """Computes and stores the average and current value""" # Docstring che descrive lo scopo della classe.

    def __init__(self): # Metodo costruttore della classe.
        self.reset() # Chiama il metodo reset() per inizializzare le variabili.

    def reset(self): # Metodo per resettare lo stato del misuratore.
        self.val = 0 # Valore corrente (valore dell'ultimo aggiornamento).
        self.avg = 0 # Valore medio (media di tutti i valori aggiornati).
        self.sum = 0 # Somma cumulativa di tutti i valori aggiornati.
        self.count = 0 # Numero di valori aggiornati.

    def update(self, val, n=1): # Metodo per aggiornare il misuratore. Prende un valore 'val' e un conteggio 'n'.
        self.val = val # Imposta il valore corrente.
        self.sum += val * n # Aggiorna la somma cumulativa.
        self.count += n # Aggiorna il conteggio totale.
        self.avg = self.sum / self.count # Ricalcola la media.


class Logger(object): # Definizione della classe Logger.

    def __init__(self, path, header): # Metodo costruttore. Prende il percorso del file e un'intestazione per il CSV.
        self.log_file = path.open('w') # Apre il file specificato in modalità scrittura ('w').
        self.logger = csv.writer(self.log_file, delimiter='\t') # Crea un oggetto writer CSV usando il file aperto, con il delimitatore di tab.

        self.logger.writerow(header) # Scrive la riga di intestazione nel file CSV.
        self.header = header # Salva l'intestazione per un uso futuro.

    def __del__(self): # Metodo distruttore, viene chiamato quando l'oggetto viene eliminato.
        self.log_file.close() # Chiude il file del log, liberando la risorsa.

    def log(self, values): # Metodo per registrare una riga di dati. Prende un dizionario 'values'.
        write_values = [] # Inizializza una lista vuota per i valori da scrivere.
        for col in self.header: # Itera attraverso le colonne definite nell'intestazione.
            assert col in values # Controlla che ogni colonna dell'intestazione sia presente nel dizionario 'values'.
            write_values.append(values[col]) # Aggiunge il valore corrispondente alla lista.

        self.logger.writerow(write_values) # Scrive la riga di valori nel file CSV.
        self.log_file.flush() # Forza la scrittura dei dati dal buffer al disco, garantendo che i log siano aggiornati immediatamente.


def calculate_accuracy(outputs, targets): # Funzione per calcolare l'accuratezza. Prende gli output del modello e i target reali.
    with torch.no_grad(): # Disabilita il calcolo dei gradienti per risparmiare memoria e tempo (utile durante la valutazione).
        batch_size = targets.size(0) # Ottiene la dimensione del batch dai target.

        _, pred = outputs.topk(1, 1, largest=True, sorted=True) # Trova l'indice della classe con il punteggio più alto. 'topk' restituisce i k valori più alti e i loro indici; qui k=1.
        pred = pred.t() # Trasforma il tensore per facilitare il confronto.
        correct = pred.eq(targets.view(1, -1)) # Confronta le predizioni con i target per vedere quali sono corrette.
        n_correct_elems = correct.float().sum().item() # Conta il numero di elementi corretti.

        return n_correct_elems / batch_size # Restituisce l'accuratezza come frazione di elementi corretti sul totale del batch.


def calculate_precision_and_recall(outputs, targets, pos_label=1): # Funzione per calcolare precisione e recall.
    with torch.no_grad(): # Disabilita il calcolo dei gradienti.
        _, pred = outputs.topk(1, 1, largest=True, sorted=True) # Ottiene le predizioni (la classe con il punteggio più alto).
        precision, recall, _, _ = precision_recall_fscore_support( # Calcola precisione, recall, f-score e supporto usando la funzione di scikit-learn.
            targets.view(-1, 1).cpu().numpy(), # Converte i target in un array NumPy sulla CPU.
            pred.cpu().numpy()) # Converte le predizioni in un array NumPy sulla CPU.

        return precision[pos_label], recall[pos_label] # Restituisce la precisione e il recall per la classe specificata da 'pos_label'.


def worker_init_fn(worker_id): # Funzione per inizializzare i worker in un DataLoader multi-processo.
    torch_seed = torch.initial_seed() # Ottiene il seed iniziale di PyTorch.

    random.seed(torch_seed + worker_id) # Imposta il seed per la libreria 'random' usando l'ID del worker per garantire un'inizializzazione diversa per ogni worker.

    if torch_seed >= 2**32: # A causa di limitazioni di NumPy, controlla se il seed è troppo grande.
        torch_seed = torch_seed % 2**32 # Se è troppo grande, lo riduce usando l'operatore modulo.
    np.random.seed(torch_seed + worker_id) # Imposta il seed per la libreria NumPy.


def get_lr(optimizer): # Funzione per ottenere il tasso di apprendimento.
    lrs = [] # Inizializza una lista vuota per i tassi di apprendimento.
    for param_group in optimizer.param_groups: # Itera attraverso i gruppi di parametri dell'ottimizzatore.
        lr = float(param_group['lr']) # Estrae il tasso di apprendimento di ogni gruppo.
        lrs.append(lr) # Aggiunge il tasso di apprendimento alla lista.

    return max(lrs) # Restituisce il tasso di apprendimento massimo trovato. Questo è utile per ottimizzatori che usano tassi diversi.


def partialclass(cls, *args, **kwargs): # Funzione decoratore per creare una "classe parziale".
    
    class PartialClass(cls): # Definisce una nuova classe che eredita dalla classe originale 'cls'.
        __init__ = partialmethod(cls.__init__, *args, **kwargs) # Sostituisce il metodo costruttore (__init__) della nuova classe con una versione "parziale" del costruttore originale. Questo significa che alcuni argomenti sono già predefiniti.

    return PartialClass # Restituisce la nuova classe parziale.