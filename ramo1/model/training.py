import torch # Importa la libreria PyTorch, essenziale per il deep learning.
import time # Importa la libreria 'time' per misurare i tempi di esecuzione.
import os # Importa la libreria 'os' per interagire con il sistema operativo (non usata direttamente in questa funzione, ma utile in genere).
import sys # Importa la libreria 'sys' (non usata direttamente in questa funzione, ma utile in genere).

import torch.distributed as dist # Importa il modulo per l'addestramento distribuito, usato per addestrare modelli su più GPU o macchine.

from utils import AverageMeter, calculate_accuracy # Importa classi e funzioni da un modulo 'utils'.
# - AverageMeter: Una classe che calcola la media e tiene traccia dei valori, usata per perdita e accuratezza.
# - calculate_accuracy: Una funzione che calcola l'accuratezza di un modello.

def train_epoch(epoch, # Numero dell'epoca corrente.
                data_loader, # L'oggetto DataLoader di PyTorch che fornisce i dati di addestramento in batch.
                model, # Il modello di rete neurale da addestrare.
                criterion, # La funzione di perdita (es. CrossEntropyLoss).
                optimizer, # L'ottimizzatore (es. Adam, SGD) per aggiornare i pesi del modello.
                device, # Il dispositivo (CPU o GPU) su cui eseguire l'addestramento.
                current_lr, # Il tasso di apprendimento corrente.
                epoch_logger, # Un oggetto per registrare i log alla fine di ogni epoca.
                batch_logger, # Un oggetto per registrare i log dopo ogni batch.
                tb_writer=None, # Oggetto per scrivere i log su TensorBoard (opzionale, valore predefinito None).
                distributed=False): # Flag booleano per indicare se l'addestramento è distribuito (opzionale, valore predefinito False).
    print('train at epoch {}'.format(epoch)) # Stampa un messaggio per indicare l'inizio dell'addestramento per l'epoca corrente.

    model.train() # Imposta il modello in modalità "train". Questo attiva funzionalità come Dropout e Batch Normalization.

    batch_time = AverageMeter() # Inizializza un misuratore per il tempo impiegato per ogni batch.
    data_time = AverageMeter() # Inizializza un misuratore per il tempo impiegato per caricare i dati.
    losses = AverageMeter() # Inizializza un misuratore per la perdita (loss).
    accuracies = AverageMeter() # Inizializza un misuratore per l'accuratezza.

    end_time = time.time() # Registra il tempo di inizio per la prima iterazione.
    for i, (inputs, targets) in enumerate(data_loader): # Itera su tutti i batch forniti da data_loader.
        data_time.update(time.time() - end_time) # Calcola e aggiorna il tempo impiegato per caricare il batch corrente.

        targets = targets.to(device, non_blocking=True) # Sposta i target (le etichette) sul dispositivo (GPU) per il calcolo. L'opzione non_blocking=True ottimizza i trasferimenti.
        outputs = model(inputs) # Esegue il forward pass: il modello riceve gli input e genera gli output (le predizioni).
        loss = criterion(outputs, targets) # Calcola la perdita confrontando gli output del modello con i target reali.
        acc = calculate_accuracy(outputs, targets) # Calcola l'accuratezza del modello sul batch corrente.

        losses.update(loss.item(), inputs.size(0)) # Aggiorna il misuratore della perdita con il valore della perdita del batch e la dimensione del batch.
        accuracies.update(acc, inputs.size(0)) # Aggiorna il misuratore dell'accuratezza con il valore dell'accuratezza del batch e la dimensione del batch.

        optimizer.zero_grad() # Azzera i gradienti accumulati dall'iterazione precedente.
        loss.backward() # Esegue il backward pass: calcola i gradienti della perdita rispetto ai pesi del modello.
        optimizer.step() # Esegue un passo dell'ottimizzatore, aggiornando i pesi del modello in base ai gradienti calcolati.

        batch_time.update(time.time() - end_time) # Calcola e aggiorna il tempo totale impiegato per l'intera iterazione del batch.
        end_time = time.time() # Aggiorna il tempo di inizio per il prossimo batch.

        if batch_logger is not None: # Controlla se è stato fornito un logger per i batch.
            batch_logger.log({ # Se sì, registra i dettagli del batch corrente.
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val, # Perdere del batch corrente.
                'acc': accuracies.val, # Accuratezza del batch corrente.
                'lr': current_lr # Tasso di apprendimento corrente.
            })

        print('Epoch: [{0}][{1}/{2}]\t' # Stampa una stringa formattata con le statistiche del batch.
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' # Tempo del batch corrente e tempo medio.
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' # Tempo di caricamento dati corrente e tempo medio.
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t' # Perdita del batch corrente e perdita media.
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, # Accuratezza del batch corrente e accuratezza media.
                                                         i + 1,
                                                         len(data_loader),
                                                         batch_time=batch_time,
                                                         data_time=data_time,
                                                         loss=losses,
                                                         acc=accuracies))

    if distributed: # Se l'addestramento è distribuito...
        loss_sum = torch.tensor([losses.sum], # Crea un tensore per la somma delle perdite.
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count], # Crea un tensore per il numero di campioni.
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum], # Crea un tensore per la somma delle accuratezze.
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count], # Crea un tensore per il numero di campioni.
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM) # Sincronizza e somma le perdite da tutti i processi distribuiti.
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM) # Sincronizza e somma il conteggio dei campioni.
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM) # Sincronizza e somma le accuratezze.
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM) # Sincronizza e somma il conteggio dei campioni.

        losses.avg = loss_sum.item() / loss_count.item() # Calcola la perdita media globale per l'epoca.
        accuracies.avg = acc_sum.item() / acc_count.item() # Calcola l'accuratezza media globale per l'epoca.

    if epoch_logger is not None: # Controlla se è stato fornito un logger per l'epoca.
        epoch_logger.log({ # Se sì, registra i dettagli dell'intera epoca.
            'epoch': epoch,
            'loss': losses.avg, # Perdita media dell'epoca.
            'acc': accuracies.avg, # Accuratezza media dell'epoca.
            'lr': current_lr # Tasso di apprendimento corrente.
        })

    if tb_writer is not None: # Controlla se è stato fornito un writer per TensorBoard.
        tb_writer.add_scalar('train/loss', losses.avg, epoch) # Scrive la perdita media su TensorBoard.
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch) # Scrive l'accuratezza media su TensorBoard.
        tb_writer.add_scalar('train/lr', accuracies.avg, epoch) # Scrive il tasso di apprendimento su TensorBoard.