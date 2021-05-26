import pyaudio  # Soundcard audio I/O access library
import wave  # Python 3 module for reading / writing simple .wav files
from Explorer import *
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

arr =[[1921, 1178, 1553, 387, 101],
      [1052, 2920, 382, 52, 939],
      [1161, 673, 2765, 1805, 192],
      [ 213, 1, 1544, 4344, 1017],
      [ 0, 0, 0, 0, 0]]

emotions = ["Happy", "Angry", "Fear", "Sad", "None"]

df_cm = pd.DataFrame(arr, index = [i for i in emotions],
                  columns = [i for i in emotions])
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
plt.show()

line = ""

for x in range(len(arr)):
    line = ""
    for y in range(len(arr[x])):
        line += str(arr[x][y]) + " "
    print(line)

#for x in range(len(arr)):
    #for y in range(len(arr[x])):
        #arr[x][y] = arr[x][y].replace('.', ',')
        #print(arr[x][y])
