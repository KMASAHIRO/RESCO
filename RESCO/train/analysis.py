import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from RESCO.config.map_config import map_configs

import xml.etree.ElementTree as ET

def read_csv(log_dir):
    names = [folder for folder in next(os.walk(log_dir))[1]]
    
    metric = 'queue'
    output_file = 'avg_{}.py'.format(metric)
    run_avg = dict()

    for name in names:
        split_name = name.split('-')
        map_name = split_name[2]
        average_per_episode = []
        for i in range(1, 10000):
            trip_file_name = log_dir+name + os.sep + 'metrics_'+str(i)+'.csv'
            if not os.path.exists(trip_file_name):
                print('No '+trip_file_name)
                break

            num_steps, total = 0, 0.0
            last_departure_time = 0
            last_depart_id = ''
            with open(trip_file_name) as fp:
                reward, wait, steps = 0, 0, 0
                for line in fp:
                    line = line.split('}')
                    queues = line[2]
                    signals = queues.split(':')
                    step_total = 0
                    for s, signal in enumerate(signals):
                        if s == 0: continue
                        queue = signal.split(',')
                        queue = int(queue[0])
                        step_total += queue
                    step_avg = step_total / len(signals)
                    total += step_avg
                    num_steps += 1

            average = total / num_steps
            average_per_episode.append(average)

        run_name = split_name[0]+' '+split_name[2]+' '+split_name[3]+' '+split_name[4]+' '+split_name[5]
        average_per_episode = np.asarray(average_per_episode)

        if run_name in run_avg:
            run_avg[run_name].append(average_per_episode)
        else:
            run_avg[run_name] = [average_per_episode]

    alg_res = []
    alg_name = []
    for run_name in run_avg:
        list_runs = run_avg[run_name]
        min_len = min([len(run) for run in list_runs])
        list_runs = [run[:min_len] for run in list_runs]
        avg_delays = np.sum(list_runs, 0)/len(list_runs)
        err = np.std(list_runs, axis=0)

        alg_name.append(run_name)
        alg_res.append(avg_delays)

        alg_name.append(run_name+'_yerr')
        alg_res.append(err)

        #plt.title(run_name)
        #plt.plot(avg_delays)
        #plt.show()


    np.set_printoptions(threshold=sys.maxsize)
    with open(output_file, 'a') as out:
        for i, res in enumerate(alg_res):
            out.write("'{}': {},\n".format(alg_name[i], res.tolist()))

def read_xml(log_dir, env_base):
    names = [folder for folder in next(os.walk(log_dir))[1]]

    metrics = ['timeLoss', 'duration', 'waitingTime']

    for metric in metrics:
        output_file = 'avg_{}.py'.format(metric)
        run_avg = dict()

        for name in names:
            split_name = name.split('-')
            map_name = split_name[2]
            average_per_episode = []
            for i in range(1, 10000):
                trip_file_name = log_dir+name + os.sep + 'tripinfo_'+str(i)+'.xml'
                if not os.path.exists(trip_file_name):
                    print('No '+trip_file_name)
                    break
                try:
                    tree = ET.parse(trip_file_name)
                    root = tree.getroot()
                    num_trips, total = 0, 0.0
                    last_departure_time = 0
                    last_depart_id = ''
                    for child in root:
                        try:
                            num_trips += 1
                            total += float(child.attrib[metric])
                            if metric == 'timeLoss':
                                total += float(child.attrib['departDelay'])
                                depart_time = float(child.attrib['depart'])
                                if depart_time > last_departure_time:
                                    last_departure_time = depart_time
                                    last_depart_id = child.attrib['id']
                        except Exception as e:
                            #raise e
                            break
                    route_file_name = env_base + map_name + os.sep + map_name + '_' + str(i) + '.rou.xml'

                    if metric == 'timeLoss':    # Calc. departure delays
                        try:
                            tree = ET.parse(route_file_name)
                        except FileNotFoundError:
                            route_file_name = env_base + map_name + os.sep + map_name + '.rou.xml'
                            tree = ET.parse(route_file_name)
                        root = tree.getroot()
                        last_departure_time = None
                        for child in root:
                            if child.attrib['id'] == last_depart_id:
                                last_departure_time = float(child.attrib['depart'])     # Get the time it was suppose to depart
                        never_departed = []
                        if last_departure_time is None: raise Exception('Wrong trip file')
                        for child in root:
                            if child.tag != 'vehicle': continue
                            depart_time = float(child.attrib['depart'])
                            if depart_time > last_departure_time:
                                never_departed.append(depart_time)
                        never_departed = np.asarray(never_departed)
                        never_departed_delay = np.sum(float(map_configs[map_name]['end_time']) - never_departed)
                        total += never_departed_delay
                        num_trips += len(never_departed)

                    average = total / num_trips
                    average_per_episode.append(average)
                except ET.ParseError as e:
                    #raise e
                    break

            run_name = split_name[0]+' '+split_name[2]+' '+split_name[3]+' '+split_name[4]+' '+split_name[5]
            average_per_episode = np.asarray(average_per_episode)

            if run_name in run_avg:
                run_avg[run_name].append(average_per_episode)
            else:
                run_avg[run_name] = [average_per_episode]


        alg_res = []
        alg_name = []
        for run_name in run_avg:
            list_runs = run_avg[run_name]
            min_len = min([len(run) for run in list_runs])
            list_runs = [run[:min_len] for run in list_runs]
            avg_delays = np.sum(list_runs, 0)/len(list_runs)
            err = np.std(list_runs, axis=0)

            alg_name.append(run_name)
            alg_res.append(avg_delays)

            alg_name.append(run_name+'_yerr')
            alg_res.append(err)

            #plt.title(run_name)
            #plt.plot(avg_delays)
            #plt.show()


        np.set_printoptions(threshold=sys.maxsize)
        with open(output_file, 'a') as out:
            for i, res in enumerate(alg_res):
                out.write("'{}': {},\n".format(alg_name[i], res.tolist()))
