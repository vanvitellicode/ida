# Implemetazione Architettura
In questa fase ciò ce è stato fatto  l'iplementazione di una demo funzionante volta a comprendere in che modo l'architettura deifinitiva dovrebbe operare. Nel dettaglio qui si discuterà di tutte le pagine che sono state implementate a partire dai framework

## Scelta del framework e Motivazioni
Come framework princiapale è stato scelto flusk. Il otivo di tale sceltà è stata dettata dal fatto che il linguaggio prnipe oe l'analisi modellazione dei dati e utilizzo di algormittmi di machine learnign è python. L sceltà di utilizzare differenti linguagi di programmazione per la gestione del frontend avrebbe complicato ultiewrmimente il priblea, quindi la solizoine più semplice è stata prorpio quella di utilizzare un framework che intregrasse un frontend a Pythion.
:In riferimento al'architetturea descritta nel documento precedente tutto il sistema di autenticazione composto da login registrazionei e relativi reindirizzamenti è stato gestitp ointramente da firebase. Sono stat relaizzati differenti tipologie di ruolo descritti in maniera dettagliata all'interno dell'overleaf. Ciascuno di quesi ha accesso ad un set limitato di impostaizonisu cui può operare.

## Pagine e schermate
Lew pagine implemtnate sono state:
- login/ regsitrazione che si occupano appunto della gestione dell'autenticazione 
- pagina principale che mi permette di effettuare le seguenti iperazioni: 
    - selezinare un dataset 
    - scegli se effettuare addestramento / usare il modello oppure testare il modello
    - a seconda di questo visualizzi risultati differenti. (Ad esempip nell'ìutilizzo ottieni direttamente le metriche di credulità o incredulità rispetto ad un post, nel test ottieni delle misurte di prestazione. eltrain ottieni il salvataggio del modello in questione.)


![Login](../11-24%20DemoWithFrontEnd/UseCasesCode/figure/login.png)

![Registrazione](../11-24%20DemoWithFrontEnd/UseCasesCode/figure/registrazione.png)


![Login](../11-24%20DemoWithFrontEnd/UseCasesCode/figure/principale.png)

![Descrizione dell'immagine](../11-24%20DemoWithFrontEnd/UseCasesCode/figure/misure.png)


![Login](../11-24%20DemoWithFrontEnd/UseCasesCode/figure/upload_dataset.png)

![Descrizione dell'immagine](../11-24%20DemoWithFrontEnd/UseCasesCode/figure/belif.png)
