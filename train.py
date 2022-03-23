from turtle import position
from models.model import SeqModel
from globals.globals import *




DEVICE = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

progress = trange(EPOCHS,desc='Epochs',leave=True)
model = SeqModel(VOCAB_SIZE)
model.to(DEVICE)

for epoch in progress:
    optimizer = Optimizer(model)
    for i,(x,y) in enumerate(train_data):
        logger.debug(f'x is {x}')
        logger.debug(f'y is {y}')
        logger.debug(f'x size before embd : {x.shape}')
        logger.debug(f'y size before embd : {y.shape}')
        logger.debug(f'x is on device : {x.device}')
        logger.debug(f'y is on device : {y.device}')
        logger.debug(f'DEVICE is : {DEVICE}')
        x = x.to(DEVICE).long()
        y = y.to(DEVICE).long()


        logger.debug(f'x is on device : {x.device}')
        logger.debug(f'y is on device : {y.device}')


        y_pred = model(x)


        # logger.debug(f'y_pred is on device : {y_pred.device}')
        # logger.debug(f'y_pred size : {y_pred.shape}')
        logger.debug(f'y size : {y.shape}')

        loss =  th.nn.CrossEntropyLoss()(y_pred,y)
        optimizer.step()
        if i%BATCH_SIZE == 0:
            progress.set_description(f"Epoch {epoch}")
            progress.set_postfix(batch=f"{int(i/BATCH_SIZE)}/{int(len(train_data)/BATCH_SIZE)}",loss=loss.detach().cpu().numpy())
        
        