# Author : CheihChiu
# Date   : 2017-06-06

import flask
import logging
from flask import request

import os.path
import sys
import time
import cnn_captcha_break
import tensorflow as tf
import numpy as np

from PIL import Image

import requests
from io import BytesIO

logger = logging.getLogger('general')
exc_logger = logging.getLogger('exception')

ICON_SET = ['a', 'b', 'c', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 't', 'u', 'x', 'y']
ICON_WIDTH = 33
ICON_HEIGHT = 70
TOTAL_NUMBER = 6
CKPT_PATH = 'application/resources/data'

code_cache = {}

def combine_letters(partition_point):
    """delete some invalid partition points
    """
    new_letters = []
    last_width = ICON_WIDTH
    last_start = -1
    for index, letter in enumerate(partition_point):
        width = letter[1] - letter[0]
        if width < 13:
            if width + last_width <= ICON_WIDTH:
                new_letters[-1] = (new_letters[-1][0], letter[1])
                last_width += width
                last_start = -1
            else:
                last_start = letter[0]
                last_width = width
        else:
            if last_start > 0:
                if width + last_width < ICON_WIDTH:
                    new_letters.append((last_start, letter[1]))
                    last_start = -1
                    last_width = width + last_width
            else:
                last_width = width
                new_letters.append(letter)
    #print("new_letters")
    #print(new_letters)
    return new_letters

def image_to_vector(im):
    im = im.convert('1')
    im_vector = np.zeros(ICON_WIDTH * ICON_HEIGHT, dtype=np.int32)
    for index, value in enumerate(im.getdata()):
        if value == 0:
            im_vector[index] = 1
    return im_vector

def vector_to_code(y):
    result = []
    for item in y:
        result.append(ICON_SET[np.argmax(item)])
    return result

def read_code_localfile(path):
    with Image.open(path) as img:
        return read_data(img)

def read_code(path):
    r = requests.get(path, timeout=30)
    with BytesIO(r.content) as f:
        with Image.open(f) as img:
            return read_data(img)

def read_data(im):
    try:
        im = im.convert("P")

        his = im.histogram()

        values = {}
        for i in range(0, 256):
            values[i] = his[i]

        # Create two white background pictures with the same size of the caphtcha image
        im_binary= Image.new("P", im.size, 255)
        im_to_crop = Image.new("P", im.size, 255)

        # Noise filtering
        for y in range(im.size[1]):
            for x in range(im.size[0]):
                pix = im.getpixel((x, y))
                if pix == 0:
                    im_binary.putpixel((x, y), 0)
                if pix < 200:
                    im_to_crop.putpixel((x, y), 0)

        inletter = False
        foundletter = False
        start = 0
        end = 0

        # the captcha can be was split lengthwise with the partition's horizontal axis 
        partition_point = []
        for x in range(im_binary.size[0]):
            for y in range(im_binary.size[1]):
                pix = im_binary.getpixel((x, y))
                if pix != 255:
                    inletter = True
                    #解决最右侧粘连的bug
                    if x == im_binary.size[0]-1 :
                        inletter = False
            if foundletter == False and inletter == True:
                foundletter = True
                start = x
                #print('fount start:%d'%start)

            if foundletter == True and inletter == False:
                foundletter = False
                end = x
                #print('start:%d,end:%d'%(start, end))
                partition_point.append((start, end))
            
            inletter = False
        #print('part count:' + str(len(partition_point)))
        if len(partition_point) > TOTAL_NUMBER:
            partition_point = combine_letters(partition_point)

        validation_images = []
        for index, partition in enumerate(partition_point):
            im_letter = im_to_crop.crop((partition[0], 0, partition[1], im_to_crop.size[1]))
            tempImage = Image.new('RGB', (ICON_WIDTH, ICON_HEIGHT), (255, 255, 255))
            startX = (ICON_WIDTH - im_letter.size[0]) // 2
            startY = (ICON_HEIGHT - im_letter.size[1]) // 2
            tempImage.paste(im_letter, (startX, startY, startX + im_letter.size[0], startY + im_letter.size[1]))
            validation_images.append(image_to_vector(tempImage))
        return validation_images
    except Exception as err:
        logger.info(err)


images_placeholder = tf.placeholder(tf.float32, shape=(TOTAL_NUMBER, ICON_WIDTH * ICON_HEIGHT))
logits = cnn_captcha_break.logits(images_placeholder, ICON_WIDTH, ICON_HEIGHT)

sess = tf.Session()
ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
if ckpt:
    print(ckpt.model_checkpoint_path)
    tf.train.Saver(tf.global_variables()).restore(sess, ckpt.model_checkpoint_path)

  

def ocrfromurl(image_path):
    try:
        image_path = 'http://106.13.231.186/amzspider-robot/1580635186089.png'
        image_path = 'http://106.13.231.186/amzspider-robot/1580653800520.png' #漏字母
        image_path = 'http://106.13.231.186/amzspider-robot/1580716407143.png'#漏字母
        
        
        path_key = image_path[image_path.rfind('/') + 1:]
        #if image_path in code_cache:
        #    return flask.jsonify({
        #        'code':  code_cache[path_key]
        #    })
        images_feed = read_code(image_path)
        
            

        y = sess.run(logits, feed_dict={images_placeholder: images_feed})
        code = "".join(vector_to_code(y))
        code_cache[path_key] = code
           
        print('code:' +   str(code))
         
        
    except Exception as err:
        logger.info(err)


def ocrfromlocal(dirpath):


    for file in os.listdir(dirpath):
        try:
            file_path = os.path.join(dirpath, file)
            filename = os.path.split(file_path)[1]

            if file_path in code_cache:
                return print(  'code:' +   code_cache[file_path])
            images_feed = read_code_localfile(file_path)

            y = sess.run(logits, feed_dict={images_placeholder: images_feed})
            code = "".join(vector_to_code(y))
            code_cache[file_path] = code
            os.rename(file_path, './codepic/'+ str(code)+'-'+str(round(time.time()))+'.png')

            print('rename '+ filename + ' ' + str(code)+'.png')
        except Exception as err:
            print('Except:' + str(err))

def main():
    ocrfromlocal('./codepic/')
main()


