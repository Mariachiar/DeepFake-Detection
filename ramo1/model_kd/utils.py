import csv # Importa la libreria 'csv' per gestire i file in formato CSV.
import random # Importa la libreria 'random' per generare numeri casuali.
from functools import partialmethod # Importa la classe 'partialmethod' dal modulo 'functools' per creare metodi parziali.

import torch # Importa la libreria PyTorch.
import numpy as np # Importa la libreria NumPy, spesso usata per lavorare con array.
from sklearn.metrics import precision_recall_fscore_support # Importa una funzione da scikit-learn per calcolare metriche di valutazione.

"""
Tensorboard logger code referenced from:
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/
Other helper functions:
https://github.com/cs230-stanford/cs230-stanford.github.io
"""

import json
import logging
import os
import shutil
import torch
from collections import OrderedDict

import tensorflow as tf
import numpy as np
import scipy.misc 
try:
    from io import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


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



class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


class Board_Logger(object):
    """Tensorboard log utility"""
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()