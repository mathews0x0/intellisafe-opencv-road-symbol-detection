# intellisafe-opencv-road-symbol-detection
driving assistant intellisafe based on HAAR opencv algorithm

based on the autorc car concept by hamuchiwa

the instream.py file sits in the raspberry pi,feeding the realtime image data to the locally connected network server
in the server ,the haar cascade classifiers are pre trained and models are developed.the server attempts to detect the presence of a trained image in the input image,if it does a box is drawn over it.
note that each classsifier would require individual invokation,a global invokation for all classsifers to be done later
