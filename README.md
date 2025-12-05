Here is the serial code for the sobel filter. Between now and 
when I give the final starter code, the sobel_filter() function is 
unlikely to change as that is the algorithm. What may change about the serial 
code is quality of life changes as well as how timing works for the serial code. 
For example STB translates pictures from color to greyscale which is technically 
a preprocessing step that should be included in the edge detection runtime.

To compile the code, use the provided Makefile

running the code takes a png input filename and output filename. For example:

```
./sobel fox.png fox-sobel.png
```

I have provided two png images but feel free to try your own png images.

As you will note, the output detects some of the lines but the accuracy is not 
particularly high. This is where adding parallelsim to the sobel filter to save
time for other preprocessing/postprocessing steps becomes beneficial. 

For now, you can play with the threashold and distance algorithm used. For the 
final starter code, I plan on including a script to test sobel filter outputs 
against ground truth as accuracy metrics are important in computer vision.

