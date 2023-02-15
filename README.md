# iaProcessImageRecognition   
   
This software used Windows library for catch screenshot of a process and for keyboard handling.   
An Unix version will be shared later.   
   
<!-- You can try it with the realease version who's published with everything you need for testing (dll and pre-trained model with links below).    -->
   
If you want to compile this with source you will need ton follow some tutorial :   
   
- Build openCV from source with CUDA if your GPU is compatible(at least +50% perf)    
- You can use pre-built version if not or compile from source without CUDA (i recommened using YOLOv3-tiny in this case if you need lot of FPS).
   
## WIP GUI :      
- https://docs.opencv.org/4.x/dc/d46/group__highgui__qt.html  


## Usable Key : 
``num pad -`` key (substract key) = Pause
``num pad +`` key (addition key) = Unpause
``Escape`` key = clean Exit programm

If you don't have num pad you can edit these keys in main.cpp with this doc :  
- https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes


### Thanks :
Icon by Freepik :
<a href="https://www.flaticon.com/free-icons/magnifying-glass" title="magnifying glass icons">Magnifying glass icons created by Freepik - Flaticon</a>