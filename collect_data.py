import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os



def keys_to_output(keys):

    output = [0,0,0]

    if 'LEFT' in keys:
        output[0] = 1
    if 'RIGHT' in keys:
        output[1] = 1
    if ' ' in keys:
        output[2] = 1
    return output


def main():
    timestamp_training_data = []
    image_training_data = []
    label_training_data = []
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    paused = False
    print('STARTING!!!')
    while(True):

        if not paused:

            keys = key_check()
            output = keys_to_output(keys)

            if (time.time()-last_time) < 0.1:
#                print ('*')
                time.sleep(.13)
                continue

#            screen = grab_screen(region=(50,50,1900,1100))
            screen = grab_screen(region=(0,0,1300,850))
#            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

            timestamp_training_data.append(int(time.time()*100000))
            image_training_data.append(screen)
            label_training_data.append(output)

#            print('loop took ' + str((time.time()-last_time)) + ' seconds')
#            print('*')
            last_time = time.time()

        keys = key_check()
        if 'T' in keys or 't' in keys:
            time.sleep(1)
            break

    print('timestamp.len', len(timestamp_training_data))
    label_file = open('data' + os.sep  + 'label_' + str(int(time.time()*100000)) + '.csv', 'w')
    label_file.write('image_filename;left;right;space')
    label_file.write('\n')

    for i in range(len(timestamp_training_data)):
        image_filename = 'img_' + str(timestamp_training_data[i]) + '.png'
        label_file.write(image_filename)
        s = ''
        for t in label_training_data[i]:
            s = s + ';' + str(t)
        label_file.write(s)
        screen = image_training_data[i]
        cv2.imwrite('data' + os.sep + image_filename, screen)
        label_file.write('\n')


main()
