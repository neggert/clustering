import matplotlib.pyplot as plt
from gm2_clustering.wf_generator import *
import math
import numpy as np
import multiprocessing

def one_vs_two(fit_func, data_file):
    """
    Test performance of an algorithm at distinguishing one-electron
    events from two-electron events.

    Parameters:
    fit_func - function that takes a single input, the waveform, and outputs a tuple
               The length of the tuple is the number of electrons the algorithm thinks
               were in the waveform.
    generate_one - function to generate a waveform from one electron
    generate_two - function to generate a waveform from two electrons

    Output:
    There are two relevant measures here.

        * The fake rate: How often the algorithm thinks there were two electrons when
        there was really only one.
        * The efficiency: How often the algorithm correctly identifies two-electron
        waveforms as a function of the distance between the two waveforms in space
        and time.
    """

    n_one_tries = 0
    n_one_correct = 0

    incorrect_dr, total_dr = [], []
    incorrect_dt, total_dt = [], []

    def num_clusters(wf):
        return len(fit_func(wf))

    def worker(in_q, out_q):
        while True:
            i = in_q.get()
            if i is None:  # poison pill
                print "Quitting worker"
                in_q.task_done()
                return
            p1, p2, wf = i
            n = num_clusters(wf)
            out_q.put((p1, p2, n))
            in_q.task_done()

    with np.load(data_file) as data:
        job_queue = multiprocessing.JoinableQueue()
        out_queue = multiprocessing.Queue()

        workers = []
        for i in xrange(8):
            w = multiprocessing.Process(target=worker, args=(job_queue, out_queue))
            w.start()
            workers.append(w)

        # Send data to workers
        n_one = 0
        for line in data['one']:
            p1, wf = line
            job_queue.put((p1, None, wf))
            n_one += 1

        # wait for workers to finish
        print "Waiting for workers to finish"
        job_queue.join()

        # retrieve the data
        print "Getting data"
        for i in xrange(n_one):
            p1, p2, n = out_queue.get()
            if n == 1:
                n_one_correct += 1
            n_one_tries += 1

        assert(out_queue.empty())

        n_two = 0
        for line in data['two']:
            p1, p2, wf = line
            job_queue.put((p1, p2, wf))
            n_two += 1

        print "Waiting for workers to finish"
        job_queue.join()


        print "Getting data"
        for i in xrange(n_two):
            p1, p2, n = out_queue.get()
            if p2 is None:
                import pdb
                pdb.set_trace()
            dr = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
            dt = abs(p1[2]-p2[2])
            if n < 2:
                incorrect_dr.append(dr)
                incorrect_dt.append(dt)
            total_dr.append(dr)
            total_dt.append(dt)

        # shut down the workers
        for i in xrange(8):
            job_queue.put(None)


    print("False-positive rate: {}".format(1-1.*n_one_correct/n_one_tries))

    total_count, bin_x, bin_y = np.histogram2d(total_dr, total_dt, bins=(8, 25))
    incorrect_count, bin_x, bin_y = np.histogram2d(incorrect_dr, incorrect_dt, bins=[bin_x, bin_y])
    ratio = 1.*incorrect_count/total_count
    ratio[np.isnan(ratio)] = 0.
    extent = [bin_y[0], bin_y[-1], bin_x[0], bin_x[-1]]
    plt.imshow(ratio, extent=extent, interpolation='nearest', aspect='auto', origin='lower')
    plt.ylabel(r"$\Delta R$ (crystals)")
    plt.xlabel(r"$\Delta t$ (ns)")
    cb = plt.colorbar()
    cb.set_label("False-negative rate")
    plt.show()
