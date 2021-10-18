import requests
import os 
import json
from threading import Lock
import time
import os 
from imageio import imread
import io
import uuid
import numpy as np
import base64
import cv2
import ast
from requests_toolbelt.multipart.encoder import MultipartEncoder
#from '/home/jetsonnano/Imagens/' import test.png

URL = "https://itdi-test-api.herokuapp.com/inference/"
#http://127.0.0.1:8000/

def addNewInference(body):
    # lock = Lock()
    # lock.acquire()

    #------------------ Save the images ------------------#

    # Convert the data to object
    # CASO USAR RABBIT MQ, DESCOMENTAR ABAIXO
    # body = ast.literal_eval(body.decode('utf-8'))

    # grab the variables
    # img_str = body['0']
    # cls_image = body['1']
    # confs = body['2']

    
    img = body[0]
    cls_image = body[1]
    confs = body[2]

    # build the image
    # img_b64 = imread(io.BytesIO(base64.b64decode(img_str)))                           
    # img = cv2.cvtColor(np.array(img_b64), cv2.COLOR_BGR2RGB)                                                                                                                                                                                                                                                                                                                                                                                                                                   
    
    # Gerate an uuid for the image
    uuidImage = uuid.uuid4().hex

    # Variables configs
    directory = '/home/jetsonnano/captured_images/'
    img_name_withoutInference = ''                                                                                                                                                                                                  
    img_name_withInference = '' 
    txt_name = ''

    #create the directory to store the images
    try:
        if not os.path.exists(directory + 'Img' + str(uuidImage)):
            os.makedirs(directory + 'Img' + str(uuidImage))
    except OSError:
        print('Error: Creating directory. ' + directory)
        return

    # Grab the path
    path = directory + 'Img' + str(uuidImage) + '/'                                                                                                                                                                                                                             

    # Save the images without yolo-inference
    img_name = path + 'img' + str(uuidImage) + '.png'
    img_name_withoutInference = 'img' + str(uuidImage) + '.png'
    txt_name = 'img' + str(uuidImage)
    cv2.imwrite(img_name, img) 

    # Write the .txt file with the classes
    f = open(path + 'img' + str(uuidImage) + ".txt", "x")
    i = 0
    for i in range (0, len(cls_image)):
        accuracy = '{:.2f}'.format(confs[i] * 100)
        f.write(str(cls_image[i]) + ' ' + str(accuracy) + '\n')
        i += 1

    f.close()
         

    # Write the image with the boxes
    img_name = path + 'imgYolo' + str(uuidImage) + '.png'
    img_name_withInference = 'imgYolo' + str(uuidImage) + '.png'
    cv2.imwrite(img_name, img)

    #----------------------------------------------------#

    #------------------ Send to the server ------------------#

    normalImage = img_name_withoutInference
    inferenceImage = img_name_withInference

    
    url = URL
    #f = open(path + txtName + '.txt', 'r')
    #inference = {f}
    #print(f.read())
    dict1 = {}

    with open(path + txt_name + '.txt', 'r') as f:
        for line in f:
            # reads each line and trims of extra the spaces 
            # and gives only the valid words
            command, description = line.strip().split(None, 1)
    
            dict1[command] = description.strip()
    
    print(dict1)
    multipart_data = MultipartEncoder(
    fields={
            "normal_image": (
                normalImage,
                open(path + normalImage, 'rb'),
                "image/png"
            ),
            "inferred_image": (
                inferenceImage,
                open(path + inferenceImage, 'rb'),
                "image/png"
            ),
            "inference": json.dumps(dict1),
            "created_at": time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())
        }
    )
    
    try:
        response = requests.post(
            url,
            headers={
                "Content-Type": multipart_data.content_type,
            },
            data=multipart_data,
        )

        print(response)

        # time.sleep(7)
        # lock.release()

        return
    except Exception as e:
            print(str(e), 'request error')

            # time.sleep(7)
            # lock.release()
            
            return
    
    
#addNewInference()
