# TensorBoardColab

A library make TensorBoard working in Colab Google 

## Install
    pip install tensorboardcolab
For auto install, please add "!pip install tensorboardcolab" at the first line of Jupyter cell

## Import
    from tensorboardcolab.utils import TensorBoardColab
    from tensorboardcolab.callbacks import TensorBoardColabCallback

## Initialization
    tbc=TensorBoardColab()  
    
## Add to Keras callback   
    model.fit(x,y,epochs=100000,callbacks=[TensorBoardColabCallback(tbc)])  
    
