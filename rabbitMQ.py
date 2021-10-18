import pika
import sys
import os
from threading import Thread
import cv2
import time
from micro_services.request import addNewInference

def main():
    print('entrou')
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost', heartbeat=600))
    channel = connection.channel()

    channel.queue_declare(queue='in'),

    def callback(img, cls_image, confs, body):
        
        Thread(target = addNewInference(body)).start()

        return

    channel.basic_consume(
        queue='in', on_message_callback=callback, auto_ack=True)

    print('rabbitMQ is running...')
    channel.start_consuming()  



if __name__ == '__main__':
    try:
        #Thread(target = main).start()
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
