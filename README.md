# Neural-Transfer
A deep learning project for transfer neural of pictures.
I heard this when I attend my lab's seminar. That was so interesting to use machine learning to
change picture's style by using another picture's style. So I found relevant literature and found some reproduced information on the Internet, and reproduced the author's method according to my own understanding.

# Introduction to each part

A more specific step-by-step description is written in the comments of the file, here is a brief description.

### load_image.py

load and show image functions.

### loss.py

In this document, I define two kinds of loss according to the theory in the paper. They are used to evaluate the style and content gap between the generated image and the original image respectively.
It is worth mentioning that the Gram matrix is used in the original text. As a novice in this field, I have not understood its principle at all... I can only know that this matrix is used to evaluate the gap in style. For those who want to learn the principles, it is recommended to read the original text of the paper and Google related knowledge.

### build_model

Using the pre-trained VGG19 model.

### run_code

This file encapsulates functions for getting result parameters and style transformations. The LBFGS optimizer is widely used in various reproduction methods, so here we also follow the methods on various websites to reproduce.

### main.py

Modify the image loading location, saving location, and training times.

# Inference
https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html

Leon A. Gatys, Alexander S. Ecker, Matthias Bethge; Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 2414-2423