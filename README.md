# mk8dx-bs-detector
Monitor the game stream and sound an alert when the blue shell is detected on the mini map.

For those of us that want a louder and more obvious alert than the in-game sound effect.

![Screenshot of the blue shell detector](https://raw.githubusercontent.com/arcprime42/mk8dx-bs-detector/main/screenshot.png)

# Program notes
It is recommended to use this program together with a 1080p video capture device similar to the Elgato HD60 X. If the video frame size is not 1920x1080, the template size will be mismatched and detection may fail.

By default, to save CPU, after a blue shell is detected the program will cease monitoring for 30 seconds and will cease drawing the screen until the next instance is found.

For advanced users, there are various settings in the source code to control the performance/CPU tradeoff. Note that the source code assumes using video capture device #0. You may need to adjust this as needed. 

# One-time setup

Requires Python 3.

```bash
python3 -m venv venv
source venv/bin/activate        # on macOS/Linux
# venv\Scripts\activate         # on Windows
pip install -r requirements.txt
```

# How to run

```bash
source venv/bin/activate        # if not already activated
# venv\Scripts\activate         # on Windows
python bs-detector.py
```

On macOS, you can prevent the computer from going to sleep with:

```bash
caffeinate -dimsum python bs-detector.py
```

Enjoy detecting all the bs!
