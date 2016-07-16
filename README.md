# The Machine
A Python program designed to emulate The Machine from [Person of Interest](http://www.imdb.com/title/tt1839578/?ref_=nv_sr_1).

### What does it do?
It uses [OpenCV](http://opencv.org/) for face recognition and accepts voice commands, with randomly selected voices used in its responses.

### Installation
##### The easy way
1. Install [WinPython](https://sourceforge.net/projects/winpython/files/)  
2. Run `pip install -r requirements.txt`  
3. Install [AVbin](http://avbin.github.io/) (for voice command mode, not required if you don't activate voice mode)
4. You're done :)  

##### The manual way
Install everything manually (see [Dependencies](https://github.com/Jo-Dan/The-Machine/blob/master/README.md#dependencies) below)

### Detailed info
##### Reddit
Join the discussion on this [thread on /r/PersonOfInterest](https://www.reddit.com/r/PersonOfInterest/comments/4suknb/the_machine_program_python/)

##### Face training
To train it to recognize your face, place photos of your face into "/facebase/1/", following the naming scheme "facebase/1/subject01.(1).jpg" where "1" and "01" are the subject number. When you run the program choose to "(r)etrain". 

Once some photos are trained you can add more using the program. Once your face is visible in the frame simply type "train as yourname" into the console. The program will then proceed to take 10 still images of your face. It can be slightly buggy at this stage so if the webcam window appears frozen simply hit enter in the console, this should allow the code to progress as normal.

Once the recognizer has been trained you can select "(l)oad" at startup. You only need to retrain if you add photos to the database

##### Designations
The program stores user info (asset number, name and designation) in the subjects.csv file, which can be edited in any spreadsheet program (or notepad, but the formatting is a little harder to understand). The designations are: ADMIN, ANALOG, USER (my current term for indigo), THREAT and UNKNOWN (my stand in for "irrelevant").

If you are an ADMIN or ANALOG you can also change a subject's designation using the command "set name as designation". For example if I was to type "set finch as admin", the program would find the subject with the name finch and set them as an admin.

##### Info
Typing the command "info" into the console (you must be an ADMIN or ANALOG to use this command) will bring up an information box on the camera stream, similar to the one in the last season.

##### Voice
If you are an ANALOG interface or an ADMIN you can type "voice" into the console, this activates the voice command mode, this essentially allows you to use any other command via speech recognition. To go back to typing simply give the command "voice" again.

The voice commands and responses do require an internet connection, however, the program runs fine without voice mode. The program saves mp3 files of each individual word it says, meaning each only needs to be downloaded once (however you may not like how the random voice says that word, deleting the mp3 will result in the voice being randomized for that word again).

##### Exit
To exit the program simply type (or say, if you are in voice mode) "exit".

### Troubleshooting

- If you get this error:
  > error: (-215) scn == 3 || scn == 4 in functioncv::ipp_cvtColor  
  
  Try increasing your webcam number until it works.  
&nbsp;
- If you get this error:
  > You'll need more than one sample to learn a model  

  Try to create a folder named "1" in the "facebase" folder and place some photos (in the "facebase\1" folder)  
&nbsp;
- If you encounter any problems not listed here, just open an issue.

### Dependencies
[Python 2.7](https://www.python.org/download/releases/2.7/) (64 bit) is required.

#### Speech

| Package/module 								| pip command 								|
| --------------------------------------------- | ----------------------------------------- |
| pyglet (requires [AVbin](http://avbin.github.io/)) 	| `pip install pyglet` 						|
| speech_recognition 							| `pip install SpeechRecognition` 			|
| natural.text 									| `pip install natural` 					|
| Pyvona 										| `pip install pyvona` 						|
| num2words 									| `pip install num2words` 					|

#### Face Recognition

| Package/module 								| pip command 								|
| --------------------------------------------- | ----------------------------------------- |
| Pillow 										| `pip install Pillow` 						|
| numpy 										| `pip install numpy` 						|

You also have to install OpenCV. You can do **one of the following**:  
1. Copy cv2.pyd (get it [from here](https://drive.google.com/file/d/0B_8BvSoNTOu6bFVZQVJ4dmxsZzQ/view?usp=sharing)) to PYTHON_INSTALLATION\Lib\site-packages ([more info](http://docs.opencv.org/3.1.0/d5/de5/tutorial_py_setup_in_windows.html))  
2. Build OpenCV with extra modules yourself ([more info](https://github.com/opencv/opencv_contrib))
