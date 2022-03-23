from globals.globals import *




class SeqModel(nn.Module):
        def __init__(self,vocab_size):
                super(SeqModel,self).__init__()

                ## For Embedding we need to pass vocabulary size as first parameter
                ## If the vector representaion for any word is having the number greater than the vocab size
                ## I'll throw index of of range error
                ###### Ex: vocabulary size for our corpus is 500
                ## ##      then the each word representation will any any number that is less than 500 like
                ####       ['this','is','example'] -> [10,45,20]

                self.embd_layer = nn.Embedding(vocab_size,50)
                self.lstm = nn.LSTM(
                                        input_size=50,
                                        hidden_size=100,
                                        num_layers=2,
                                        batch_first=True,
                                        # dropout=0.5
                                )
                self.fc = nn.Linear(100,1)
                self.dropout = nn.Dropout(0.2)
                self.relu = nn.ReLU()
                self.leky = nn.LeakyReLU()
                self.fl = nn.Sigmoid()
        def forward(self,x):
                logger.debug(f'x size before embd : {x.shape}')
                x = self.embd_layer(x)
                logger.debug(f'x size after embd : {x.shape}')
                x = self.dropout(x)
                x_out, (x_hid,x_cell) = self.lstm(x)

                logger.debug(f'x size after lstm : {x.shape}')

                x = self.fc(x_hid[-1])

                logger.debug(f'x size after fc : {x.shape}')
                logger.debug(f'BATCH_SIZE : {BATCH_SIZE}')
                
                logger.debug(f'x size after view : {x.shape}')
                x = self.relu(x)
                # x = self.fl(x)
                logger.debug(f'x size after fl : {x.shape}')
                return x


