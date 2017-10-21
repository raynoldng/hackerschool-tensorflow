import pygame
from pygame.locals import *
import sys, os
import time
from PIL import Image, ImageFilter

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import cv2

pygame.init()
mouse = pygame.mouse
fpsClock = pygame.time.Clock()
width = 280
height = 280

window = pygame.display.set_mode((width, height))
canvas = window.copy()

#                     R    G    B
BLACK = pygame.Color( 0 ,  0 ,  0 )
WHITE = pygame.Color(255, 255, 255)

pygame.display.set_caption('Paintme')


# tensor flow stuff

def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
    
    tv = list(newImage.getdata()) #get pixel values
    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    return tva

sess = tf.Session()

x = tf.placeholder(tf.float32, [None, 784])
# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

# First Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout Layer
keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
h_fc1_drop = h_fc1

# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
predict_op = tf.argmax(y_conv, 1)

saver = tf.train.Saver()
model_path = "./tmp/model.ckpt"
# saver.restore(sess, model_path)
# print("model restored")

# check out meta_signature_graph

def predict(image_path):
    # im = imageprepare(image_path)
    # print(type(im))
    im = cv2.imread("screenshot.png")
    # print(im.shape)
    temp = im.reshape(28*28, 3)
    im = [1-i[0]/255 for i in temp]

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        result = sess.run(predict_op, {x: [im]})
        return result


def resize(img_path):
    im = Image.open(img_path)
    im = im.resize((28, 28))
    im.save(img_path, "png")

canvas.fill(WHITE)
window.blit(canvas, (0, 0))
window.fill(WHITE)

myfont = pygame.font.SysFont("monospace", 16)
predicted_digit = -1
scoretext = myfont.render("Predicted: " + str(predicted_digit), 1, (0, 0, 0))
window.blit(scoretext, (5, 5))

while True:
    for event in pygame.event.get():
        left_pressed, middle_pressed, right_pressed = pygame.mouse.get_pressed()
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif left_pressed:
            pygame.draw.rect(window, BLACK, pygame.mouse.get_pos()+(10,10),6)
            # pygame.draw.circle(window, BLACK, (pygame.mouse.get_pos()), 10)
        elif right_pressed:
            print("saving image")
            image_path = "screenshot.png"
            pygame.image.save(window, image_path)
            resize(image_path)
            # erase text
            scoretext = myfont.render("Predicted: " + str(predicted_digit), 1, (255, 255, 255))
            window.blit(scoretext, (5, 5))
            predicted_digit = predict(image_path)
            scoretext = myfont.render("Predicted: " + str(predicted_digit), 1, (0, 0, 0))
            window.blit(scoretext, (5, 5))


        elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
            window.fill(WHITE)
            scoretext = myfont.render("Predicted: " + str(predicted_digit), 1, (0, 0, 0))
            window.blit(scoretext, (5, 5))
        pygame.display.update()

