# mk8dx-bs-detector
Monitor the game stream and sound an alert when the blue shell is detected on the mini map.

For those of us that want a louder and more obvious alert than the in-game sound effect.

![alt text](https://raw.githubusercontent.com/arcprime42/mk8dx-bs-detector/main/screenshot.png)

# Program notes
It is recommended to use this program together with a 1080p video capture device similar to the Elgato HD60 X.

If the video frame size is not 1920x1080, then the template size will be mismatched and will fail to produce an alert.

After a blue shell is detected, the program will cease monitoring for 30 seconds to save CPU.

After a blue shell is detected, the program will cease drawing the screen until the next instance is found.

For advanced users, there are various settings in the source code to control the performance/CPU tradeoff. 

Note that the source code assumes using video capture device #0. You may need to adjust this as needed. 

# How to install dependencies

Go to terminal and type: 
* `pip install opencv-python`
* `pip install playsound`
* `pip install numpy`

Note that your system may have `python` or `python3` (the latter especially on MacOS).  
Replace `pip` with `pip3` as needed. Replace `python` with `python3` as needed.

# How to run
Go to terminal and type:

* `python bs-detector.py`

Enjoy detecting all the bs!
