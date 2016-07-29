"""
Person of Interest : The Machine

By Jo-dan
"""
import sys
import time
import timeit
import csv
import glob
import os
import threading
import cv2
from PIL import Image
import numpy as np
import Queue
# import requests

# my functions
import faceframes
from voicecontrol import get_mp3, get_speech, get_nato

# ==============================OPTIONS====================================== #
# =========================================================================== #
# webcam number
Camera_Number = raw_input("Camera Number >>> ")
vc = cv2.VideoCapture(int(Camera_Number))

# paths
face_database = 'facebase'
cascadepath = "haarcascade_frontalface_default.xml"
facecascade = cv2.CascadeClassifier(cascadepath)
log_file = open('Machine_log.txt', 'a')
subject_types = ['ADMIN', 'ANALOG', 'THREAT', 'UNKNOWN', 'USER']

#image borders
top_border = 150
side_border = 250

# colours
admin_colour = (255, 000, 000)
analog_colour = admin_colour
user_colour = (58, 238, 247)
unknown_colour = (000, 000, 255)
threat_colour = (000, 000, 255)
back_colour = (255, 255, 255)

# font of text on video
font = cv2.FONT_HERSHEY_SIMPLEX
recognizer = cv2.face.createLBPHFaceRecognizer()
# =========================================================================== #
# =========================================================================== #

# Set Print to flush
class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout = Unbuffered(sys.stdout)

# load subject database
timenow = time.strftime("%d/%m/%Y") + ' - ' + time.strftime("%I:%M:%S")
log_file.write('\n\nInitialised at {}. \n'.format(timenow))
with open('subjects.csv', "rb") as subjects:
    reader = csv.reader(subjects)
    subject_name = []
    subject_type = []
    for row in reader:
        subject_name.append(row[1])
        if len(row[2]) == 0 or row[2].upper() not in subject_types:
            subject_type.append('UNKNOWN')
        else:
            subject_type.append(row[2])
    subject_name = [x.upper() for x in subject_name]
    subject_type = [x.upper() for x in subject_type]
    log_file.write('   CSV Read. \n')
    subject_type[0] = "UNKNOWN"
    subject_name[0] = "UNKNOWN"


def rewrite_csv():
    """updates csv using subject_name and subject_type"""
    with open('subjects.csv', "wb") as subjects:
        writer = csv.writer(subjects)
        for x in range(len(subject_type)):
            if x == 0:
                writer.writerow(['Subject No.', 'Name',
                                 'Type (ADMIN/USER/THREAT/ANALOG/UNKNOWN)'])
            else:
                row = [x, subject_name[x], subject_type[x]]
                writer.writerow(row)


def normal_subject_path(a):
    """ Returns path of "subject.normal" image file"""
    normface = face_database + "/subject{}.normal".format(str(a).zfill(2))
    normopen = Image.open(normface)
    normnp = np.array(normopen, 'uint8')
    return normnp

def get_images_and_labels(path, show=True):
    """ Returns lists of images and their labels"""
    image_paths = glob.glob(path + "\\*\\*")
    images = []
    labels = []
    print "Training faces",
    for image_path in image_paths:
        print ". ",
        #read and make grey
        image_pil = Image.open(image_path).convert('L')
        #convert img to numpy array
        image = np.array(image_pil, 'uint8')
        #get image label
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        #detect faces in image
        faces = facecascade.detectMultiScale(image)
        #if face detected append face to images and label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            if show:
                cv2.imshow("Adding faces to training set...", image[y: y + h, x: x + w])
                cv2.waitKey(1)
    # return the images and labels lists
    print "\nTraining complete"
    return images, labels

def database_load(retrain=False):
    """ Rebuilds or loads face database"""
    if retrain:
        showloaded = raw_input("Display images loaded? (slower) (y/n) >>> ")
        if showloaded == 'y':
            images, labels = get_images_and_labels(face_database)
            cv2.destroyAllWindows()
        else:
            images, labels = get_images_and_labels(face_database, False)

        recognizer.train(images, np.array(labels))
        recognizer.save('trainingsaved')
        log_file.write('   Recognizer Retrained \n')
    elif not retrain:
        recognizer.load('trainingsaved')
        log_file.write('   Recogniser Training Loaded \n')

    cv2.destroyAllWindows()
#shape_type = raw_input("(b)oxes, (c)circles, poi (o)verlay,\
# samaritan (so)overlay, (p)oi or (s)amaritan? >>> ")
shape_type = 'o'
log_file.write('   Run in "{}" Mode. \n'.format(shape_type))

database = raw_input("Would you like to (r)ebuild, or (l)oad the database? >>> ")
if database.upper() == 'R':
    database_load(True)
elif database.upper() == 'L':
    database_load(False)
else:
    print "Unknown command, loading existing database..."
    database_load(False)


if shape_type == 'p' or shape_type == 'o':
    admin_colour = (58, 238, 247)
    analog_colour = (58, 238, 247)
    user_colour = (243, 124, 13)
    unknown_colour = (255, 255, 255)
    threat_colour = (000, 000, 255)
    back_colour = (000, 000, 000)

q = Queue.Queue()
q2 = Queue.Queue()

def facerec():
    """ Face recognition and video stream"""
    nbr_replacement = []
    nbr_old = [-1]
    nbr_predicted = 0
    display_infobox = False
    display_status = False
    present = 'unknown'
    exitprog = False
    accesstext = False
    starttime = int(timeit.default_timer())

    while True:
        # read frame by frame
        ret, frame_nobord = vc.read()
        frame = cv2.copyMakeBorder(frame_nobord, top_border, top_border,
                                   side_border, side_border, cv2.BORDER_CONSTANT,
                                   (0, 0, 0, 0))
        admin_present = False
        user_present = False
        unknown_present = False
        threat_present = False
        analog_present = False
        try:
            grey_predict = cv2.cvtColor(frame_nobord, cv2.COLOR_BGR2GRAY)
        except:
            print "No camera stream found, exit the program and try another camera number"
            break
        predict_image = np.array(grey_predict, 'uint8')
        faces = facecascade.detectMultiScale(predict_image, 1.03, 5, 0, (150, 150))
        for (x, y, w, h) in faces:
            nbr_predicted, conf = recognizer.predict(predict_image[y:y+h, x:x+w])
            x = x + side_border
            y = y + top_border
            if conf > 50:
                nbr_predicted = 0

            # strings for stream
#            subtxt = "Subject: {}".format(nbr_predicted)
#            nametxt = "Name: {}".format(subject_name[nbr_predicted])
#            typetxt = "Type: {}".format(subject_type[nbr_predicted])

            # Text on stream
            if subject_type[nbr_predicted] == 'ADMIN':
                all_colour = admin_colour
                admin_present = True
            elif subject_type[nbr_predicted] == 'USER':
                all_colour = user_colour
                user_present = True
            elif subject_type[nbr_predicted] == 'UNKNOWN':
                all_colour = unknown_colour
                unknown_present = True
            elif subject_type[nbr_predicted] == "THREAT":
                all_colour = threat_colour
                threat_present = True
            elif subject_type[nbr_predicted] == "ANALOG":
                all_colour = analog_colour
                analog_present = True

            if shape_type == 'o':
                frame = faceframes.poi_image(frame, x, y, w, h,
                                             subject_type[nbr_predicted])
                if display_infobox:
                    frame = faceframes.poi_infobox(frame, x+w+30, y+int(h*.5-50), nbr_predicted,
                                                   subject_name[nbr_predicted], subject_type[nbr_predicted])
                    
                    
#                subco = (x-20, y+h+45)
#                nameco = (x-20, y+h+70)
#                typeco = (x-20, y+h+95)
#            elif shape_type == 'c':
#                cv2.circle(frame, (x+int(round(.5*w)), y+int(round(.5*h))),
#                           int(round(.6*h)), all_colour, 4)
#                subco = (x+w+30, y+int(round(.5*h))-25)
#                nameco = (x+w+30, y+int(round(.5*h)))
#                typeco = (x+w+30, y+int(round(.5*h))+25)
#            elif shape_type == 'p':
#                faceframes.poi_box(frame, x, y, w, h,
#                                   subject_type[nbr_predicted])
#                subco = (x, y+h+25)
#                nameco = (x, y+h+50)
#                typeco = (x, y+h+75)
#            elif shape_type == 's':
#                faceframes.sam_circle(frame, x, y, w, h,
#                                      subject_type[nbr_predicted])
#                subco = (x+w+30, y+int(round(.5*h))-25)
#                nameco = (x+w+30, y+int(round(.5*h)))
#                typeco = (x+w+30, y+int(round(.5*h))+25)
#            
#            elif shape_type == 'so':
#                frame = faceframes.samaritan_image(frame, x, y, w, h,
#                                                   subject_type[nbr_predicted])
#                subco = (x+w+30, y+int(round(.5*h))-25)
#                nameco = (x+w+30, y+int(round(.5*h)))
#                typeco = (x+w+30, y+int(round(.5*h))+25)
#            else:
#                cv2.rectangle(frame, (x, y), (x+w, y+h), all_colour, 2)
#                subco = (x, y+h+25)
#                nameco = (x, y+h+50)
#                typeco = (x, y+h+75)
#            if not display_infobox:
#                cv2.putText(frame, subtxt, subco, font, .7, back_colour, 3)
#                cv2.putText(frame, subtxt, subco, font, .7, all_colour, 2)
#                cv2.putText(frame, nametxt, nameco, font, .7, back_colour, 3)
#                cv2.putText(frame, nametxt, nameco, font, .7, all_colour, 2)
#                cv2.putText(frame, typetxt, typeco, font, .7, back_colour, 3)
#                cv2.putText(frame, typetxt, typeco, font, .7, all_colour, 2)

            if nbr_predicted not in nbr_old:
                if nbr_predicted != 0:
                    #print "Recognized as {} ({}). (Confidence : {})".format(nbr_predicted,
                    #                                                        subject_name[nbr_predicted], conf)
                    # requests.post("https://maker.ifttt.com/trigger/Face_Detected/with/key/d0reP2BKasF7WXr86DXIxq",
                    #                data={"value1":subject_type[nbr_predicted],
                    #                      "value2":subject_name[nbr_predicted],
                    #                      "value3":str(Camera_Number)})
                    log_file.write('      Subject {} recognised:  {} \n'.format(nbr_predicted,
                                                                                subject_type[nbr_predicted]))
                else:
                    #print "Unrecognised face"
                    log_file.write('      Unrecognised face detected\n')

                #recognp = normal_subject_path(nbr_predicted)
                #cv2.imshow("Recognised as...", recognp)
                # oldnp = normal_subject_path(nbr_old)
                # cv2.imshow("Previous", oldnp)
                nbr_replacement.append(nbr_predicted)
                nbr_old = list(nbr_replacement)
        if len(nbr_old) != 0 and len(faces) == 0:
            #print 'No face in frame.'
            del nbr_old[:]
        del nbr_replacement[:]


        if threat_present:
            if accesstext:
                cv2.putText(frame, 'THREAT DETECTED', (5, 25),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'ACCESS: DENIED', (5, 55),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'THREAT DETECTED', (5, 25),
                            font, 1, threat_colour, 2)
                cv2.putText(frame, 'ACCESS: DENIED', (5, 55),
                            font, 1, threat_colour, 2)
            present = 'threat'
        elif analog_present:
            if accesstext:
                cv2.putText(frame, 'ANALOG INTERFACE DETECTED', (5, 25),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'ACCESS: GRANTED', (5, 55),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'ANALOG INTERFACE DETECTED', (5, 25),
                            font, 1, analog_colour, 2)
                cv2.putText(frame, 'ACCESS: GRANTED', (5, 55),
                            font, 1, analog_colour, 2)
            present = 'analog'
        elif admin_present:
            if accesstext:
                cv2.putText(frame, 'ADMIN DETECTED', (5, 25),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'ACCESS: GRANTED', (5, 55),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'ADMIN DETECTED', (5, 25),
                            font, 1, admin_colour, 2)
                cv2.putText(frame, 'ACCESS: GRANTED', (5, 55),
                            font, 1, admin_colour, 2)
            present = 'admin'
        elif user_present:
            if accesstext:
                cv2.putText(frame, 'USER DETECTED', (5, 25),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'ACCESS: RESTRICTED', (5, 55),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'USER DETECTED', (5, 25),
                            font, 1, user_colour, 2)
                cv2.putText(frame, 'ACCESS: RESTRICTED', (5, 55),
                            font, 1, user_colour, 2)
            present = 'user'
        elif unknown_present:
            if accesstext:
                cv2.putText(frame, 'UNKNOWN USER', (5, 25),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'ACCESS: DENIED', (5, 55),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'UNKNOWN USER', (5, 25),
                            font, 1, unknown_colour, 2)
                cv2.putText(frame, 'ACCESS: DENIED', (5, 55),
                            font, 1, unknown_colour, 2)
            present = 'unknown'

        vcheight, vcwidth = frame.shape[:2]
        cv2.putText(frame, 'Camera ' + str(Camera_Number),
                    (0, vcheight - 10), font, 1, (0, 0, 0), 4)
        cv2.putText(frame, 'Camera ' + str(Camera_Number),
                    (0, vcheight - 10), font, 1, (255, 255, 255), 1)
        
        stoptime = int(timeit.default_timer())
        uptimesec = stoptime - starttime
        if uptimesec > 59:
            mins, secs = divmod(round(uptimesec), 60)
            if mins > 59:
                hrs, mins = divmod(mins, 60)
                if hrs >= 24:
                    days, hrs = divmod(hrs, 24)
                    if days != 1:
                        uptime = "{} DAYS, {} HOURS".format(int(days), int(hrs))
                    else:
                        uptime = "1 DAY, {} HOURS".format(int(hrs))
                else:
                    if hrs != 1:
                        uptime = "{} HOURS, {} MINUTES".format(int(hrs), int(mins))
                    else:
                        uptime = "1 HOUR, {} MINUTES".format(int(mins))
            else:
                if mins != 1:
                    uptime = "{} MINUTES, {} SECONDS".format(int(mins), int(secs))
                else:
                    uptime = "1 MINUTE, {} SECONDS".format(int(secs))
        else:
            uptime = "{} SECONDS".format(int(uptimesec))
            
        if display_status:
                    frame = faceframes.poi_statusbox(frame, 0, vcheight - 150, uptime, len(faces))
        
        q.put(present)
        if not q2.empty():
            queuein = q2.get(block=False)
            if queuein == 'info':
                if not display_infobox:
                    display_infobox = True
                else:
                    display_infobox = False
            elif queuein == 'status':
                if not display_status:
                    display_status = True
                else:
                    display_status = False
                
            elif 'train' in queuein:
                train_as = queuein.replace("train as ", "")
                print "Training as {}".format(train_as)
                new_name = train_as.upper()
                if new_name in subject_name:
                    new_user_num = str(subject_name.index(new_name)).zfill(2)
                    newuser = False
                else:
                    new_user_num = str(len(subject_type)).zfill(2)
                    subject_name.append(new_name)
                    subject_type.append('UNKNOWN')
                    newuser = True
                for frameno in range(10):
                    frame_save_path = '{}\\{}\\subject{}.jpg'.format(face_database,
                                                                     int(new_user_num), new_user_num)
                    ret2, train_frame = vc.read()
                    cv2.imwrite(frame_save_path, train_frame)
                    if frameno == 0 and newuser:
                        os.rename(frame_save_path, frame_save_path.replace('jpg', "normal"))
                    else:
                        file_number = len(glob.glob(face_database + "\\{}\\*".format(int(new_user_num)))) + 2
                        sys.stdout.flush()
                        os.rename(frame_save_path, frame_save_path.replace('jpg', "({}).jpg".format(file_number)))
                    print "Captured frame {} of 10".format(frameno)

                    grey_predict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    predict_image = np.array(grey_predict, 'uint8')
                    faces = facecascade.detectMultiScale(predict_image, 1.03, 5, 0, (150, 150))
                    for (x, y, w, h) in faces:
                        cv2.rectangle(train_frame, (x, y), (x+w, y+h), all_colour, 2)
                        cv2.putText(train_frame, "Training as {}. ({} of 10)".format(new_user_num, frameno), (5, 25),
                                    font, 1, (0, 0, 0), 2)
                    cv2.imshow('stream', train_frame)
                    cv2.waitKey(500)
                database_load(True)
                log_file.write('   Recogniser Training Loaded \n')
                print "Training Complete. {} set as {}".format(subject_name[int(new_user_num)],
                                                               subject_type[int(new_user_num)])
                q.put('trained')
            elif queuein == 'exit':
                print 'exiting'
                exitprog = True

        cv2.imshow('stream', frame)
        wait = cv2.waitKey(1)
        if wait == 27 or exitprog:
            vc.release()
            cv2.destroyAllWindows()
            break


def commands():
    """ Console and voice command system"""
    #commandlist = ['info', 'set', 'names', 'train', 'voice', 'exit']
    #asset_types = ['ADMIN', 'ANALOG', 'USER', 'UNKNOWN', 'THREAT']
    vocal_input = False
    print "What are your commands?"
    while True:
        if vocal_input:
            user_input = get_speech()
        else:
            user_input = raw_input('>>> ')
        if not q.empty():
            while not q.empty:
                q.get()
            time.sleep(.001)
            present = q.get(block=False)
            if 'retrain' in user_input:
                database_load(True)
            if 'train' in user_input:
                while not q.empty():
                    q.get()
                q2.put(user_input)
                while q.get() != 'trained':
                    time.sleep(0.01)
            elif 'exit' in user_input:
                q2.put('exit')
                break
            if present == 'threat':
                print 'Threat detected. Taking precautions. Shutdown imminent'
                q2.put('exit')
                break
            elif present == 'analog' or present == 'admin':
                if user_input == 'info':
                    q2.put('info')
                elif user_input == 'status':
                    q2.put('status')
                elif 'set' in user_input:
                    set_comm = user_input.replace('set ', "").split(' as ')
                    if set_comm[1].upper() in subject_types:
                        try:
                            int_set_comm = int(set_comm[0])
                            subject_type[int(set_comm[0])] = set_comm[1].upper()
                            print "Subject {} ({}) set as {}".format(set_comm[0], subject_name[int_set_comm],
                                                                     subject_type[int(set_comm[0])])
                        except ValueError:
                            try:
                                upper_name = set_comm[0].upper()
                                subject_type[subject_name.index(upper_name)] = set_comm[1].upper()
                                print "Subject {} ({}) set as {}".format(subject_name.index(upper_name),
                                                                         upper_name, subject_type[subject_name.index(upper_name)])
                            except ValueError, e:
                                print str(e)
                                print "Name not found"
                    else:
                        print "Invalid designation"
                elif 'names' in user_input:
                    if vocal_input:
                        namelist = ""
                        for name in subject_name[1:]:
                            namelist += get_nato(name) + ";"
                        get_mp3(namelist[:len(namelist) - 1])
                    else:
                        print subject_name[1:]
                        #print str(subject_name[1:]).replace(',', ';').replace("[", "").replace("]","").replace("'","")
                elif 'voice' in user_input:
                    if not vocal_input:
                        get_mp3('Can you hear me?')
                        confirmation = get_speech()
                        for yes in ['yes', 'absolutely', 'yeah']:
                            if yes in confirmation:
                                vocal_input = True
                                get_mp3('good ; analog interface enabled')
                        if not vocal_input:
                            get_mp3('analog interface not detected ; voice commands disabled')
                    else:
                        get_mp3('analog interface disabled')
                        vocal_input = False
                else:
                    print 'Unknown command'
            elif present == 'user':
                print "Unauthorized user or command unknown."
            elif present == 'unknown':
                print 'Unknown subject detected. Access Denied'
        else:
            print "No face detected"

recog = threading.Thread(target=facerec)
recog.setDaemon(True)
recog.start()
commands()
recog.join()
rewrite_csv()

timenow = time.strftime("%d/%m/%Y") + ' - ' + time.strftime("%I:%M:%S")
log_file.write('Program Terminated at {}. \n'.format(timenow))
log_file.close()
print '\n\n.......\nGoodbye \n.......'
time.sleep(2.5)
