# transformer-from-scratch-using-numpy

Functions in the Transformer architecture, rewritten from scratch using numpy.

These are basically the small rectangles in the Transformer diagram:
<img width="557" alt="image" src="https://user-images.githubusercontent.com/80630045/236146751-b326edbf-c4be-44f9-8811-c16ddd2dd59f.png">

<img width="758" alt="image" src="https://user-images.githubusercontent.com/80630045/236146854-17bf0a93-83b7-4838-8b90-d444597132d5.png">

DISCLAIMER: these functions are basically unusable for pytorch, tensorflow, and other automatic differentiation libraries for training. 
The training would need to be coded by hand, making these functions impractical for real use.
These are just for a better, in-depth showcase of how specific parts of the transformer model works.

Works referenced: 

https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

https://github.com/ajhalthor/Transformer-Neural-Network/blob/main/transformer.py
