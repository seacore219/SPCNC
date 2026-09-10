#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('SNN_2input_inputcurrents.txt', sep='\s+')

print(df)

fig, ax = plt.subplots()
ax.plot(df["time"]/1.0e-9, df["I(L18)"]/1.0e-6, label="input 1")
ax.plot(df["time"]/1.0e-9, df["I(L19)"]/1.0e-6, label="input 2")
ax.plot(df["time"]/1.0e-9, df["I(L18)+I(L19)"]/1.0e-6, label="input 1+2")
ax.plot(df["time"]/1.0e-9, df["I(R2)"]/1.0e-6, label="loop")

ax.set(xlabel='time [ns]', ylabel='current [uA]',
       title='transponder input')
ax.grid()
ax.set_xlim(5,20)

plt.legend()

fig.savefig("test.png", bbox_inches="tight")
#plt.show()



