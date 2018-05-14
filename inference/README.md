# keras-openpose-reproduce

Sample codes for inference.


## Download model

Please download the trained model, and put it in this folder. [Dropbox](https://www.dropbox.com/s/76b3r8rj82wicik/weights.0100.h5?dl=0)


## Running inference

Please run the following command, and you will get the prediction result `result.jpg`

    $ python3 prediction.py

Due to keras model initialization, the prediction may be slow for the very first input images.