# TensorBoardColab

A library make TensorBoard working in Colab Google

## Install

    pip install tensorboardcolab

In Colab Google Jupyter, for auto install and ensure using latest version of TensorBoardColab, please add "!pip install -U tensorboardcolab" at the first line of Jupyter cell

## Import

    from tensorboardcolab import *

## Initialization

    tbc=TensorBoardColab()

After initialization, TensorBoard link will be shown in Colab Google Juyter output

PS: If Initialization failed and keep retrying forever, please increase startup_waiting_time larger than 8 seconds as below

    tbc=TensorBoardColab(startup_waiting_time=30)

## Add to Keras callback

    model.fit(x,y,epochs=100000,callbacks=[TensorBoardColabCallback(tbc)])

## Save picture to TensorBoard

    tbc.save_image(title="test_title", image=image)

## Save a value to graph of TensorBoard

    tbc.save_value("graph_name", "line_name", epoch, value)
    .
    .
    .
    tbc.flush_line(line_name)
    tbc.close()

## Thanks

ngrok !
