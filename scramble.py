import os
import random 

def scramble():
    path1 = "waveforms6/"
    path2 = "waveform_mix/"
    path3 = "test_waveforms6/"
    folders = ['country', 'hiphop_pop', 'classical', 'rock', 'jazz']

    for x in folders:
        command = "mv "+ path1 + x+ "/* " + path2 + x
        command2 = "mv "+ path3 + x+ "/* " + path2 + x
        os.system(command)
        os.system(command2)

    for x in folders:
        l = os.listdir(path2 + x)
        num = len(l)
        twenty = int(num*.2)
        to_pull = random.sample(range(num), twenty)
        for filename in to_pull:
            command = "mv " + path2 + x + "/"+ l[filename] + " " + path3 + x 
            os.system(command)
        command = "mv " + path2 + x + "/* " + path1 + x
        os.system(command)