# Spiking Transponder

## Introduction
LTspice simulations of spiking transponder network.

### Linux setup

On linux you can download LTspice for wine64
https://ltspice.analog.com/software/LTspiceIV.exe

LTspice launcher script:
```bash
#!/bin/bash
wine $HOME/.wine-appimage-stable/drive_c/Program\ Files\ \(x86\)/LTC/LTspiceIV/scad3.exe "$@"
```
In the project directory I had to link the ltspice component library
```
ln -s '/home/whit/.wine-appimage-stable/drive_c/Program Files (x86)/LTC/LTspiceIV' ltspice
ln -s '/home/whit/.wine-appimage-stable/drive_c/Program Files (x86)/LTC/LTspiceIV/lib' lib
```
Only the second link was really needed. 

 
 
## `SNN_2input.asc` 

Two pulse generators feeding into two transponders which couple to a third transponder.


 
 
