### GTSAM Installation
#### No python wrapper 
```
cd external
git clone https://github.com/borglab/gtsam -b 4.2
cd gtsam
mkdir build && cmake .. && sudo make install 
```

### GPMP2 Installation
```
cd external
git clone https://github.com/borglab/gpmp2
cd gpmp2
mkdir build && cmake .. && sudo make install
```

### GPP Installation
```
cd navigation/src
git clone xxx
cd ..
colcon build
```