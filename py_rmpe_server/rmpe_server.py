#!/usr/bin/env python
import sys
import numpy as np
import zmq
from multiprocessing import Process
from time import time

sys.path.append("..")

from py_rmpe_data_iterator import RawDataIterator


class Server:

    # these methods all called in parent process

    def __init__(self, h5file, port, name, shuffle, augment):

        self.name = name
        self.port = port
        self.h5file = h5file

        self.shuffle = shuffle
        self.augment = augment

        self.process = Process(target=Server.loop, args=(self,))
        self.process.daemon = True
        self.process.start()



    def join(self):

        return self.process.join(10)

    # these methods all called in child process

    def init(self):

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.set_hwm(1) #TODO: put 160 in production, made for debug purposes
        self.socket.bind("tcp://*:%s" % self.port)

    @staticmethod
    def loop(self):

        print("%s: Child process init... " % self.name)
        self.init()

        iterator = RawDataIterator(self.h5file, shuffle=self.shuffle, augment=self.augment)

        print("%s: Loop started... " % self.name)

        num = 0
        generation = 0
        cycle_start = time()

        while True:

            keys = iterator.num_keys()
            print("%s: generation %s, %d images " % (self.name, generation, keys))

            start = time()
            for (image, mask, labels, keypoints) in iterator.gen():

                augment_time = time()-start

                headers = self.produce_headers(image, mask, labels, keypoints)
                self.socket.send_json(headers)
                self.socket.send(np.ascontiguousarray(image))
                self.socket.send(np.ascontiguousarray(mask))
                self.socket.send(np.ascontiguousarray(labels))
                self.socket.send(np.ascontiguousarray(keypoints))

                num += 1
                print("%s [%d/%d] aug %0.2f ms (%0.2f im/s), send %0.2f s" % (self.name, num, keys, augment_time*1000, 1./augment_time,  time() - start - augment_time) )
                start = time()

    def produce_headers(self, img, mask, labels, keypoints):

        header_data = {"descr": img.dtype.str, "shape": img.shape, "fortran_order": False}
        header_mask = {"descr": mask.dtype.str, "shape": mask.shape,   "fortran_order": False}
        header_label = {"descr": labels.dtype.str,  "shape": labels.shape, "fortran_order": False}
        header_keypoints = {"descr": keypoints.dtype.str,  "shape": keypoints.shape, "fortran_order": False}

        headers = [header_data, header_mask, header_label, header_keypoints]

        return headers


def main():

    train = Server("../dataset/train_dataset.h5", 5555, "Train", shuffle=False, augment=True)
    val = Server("../dataset/val_dataset.h5", 5556, "Val", shuffle=False, augment=False)

    processes = [val, train]

    while None in [p.process.exitcode for p in processes]:

        print("exitcodes", [p.process.exitcode for p in processes])
        for p in processes:
            if p.process.exitcode is None:
                p.join()


np.set_printoptions(precision=1, linewidth=100*3, suppress=True, threshold=100000)
main()


