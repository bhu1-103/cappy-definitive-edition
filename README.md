# cappy-definitive-edition

`apt update; apt install openjdk-21-jdk neovim zsh git python3-pandas`

final

use the 3 scripts to automate...

replicate.sh creates multiple folders

mega-clean.sh cleans all redundant files/folders

4-14.sh is a sample file to automate

input features  : `node_code wlan_code x(m) y(m) z(m)`

output features : `throughput` (for now)


## [STEP 1](https://github.com/bhu1-103/cappy-definitive-edition/tree/main/step1)
### Generate randomized input parameters
 - Used java code from ITU ML challenge for generating. `javac *.java; java x-y`
    - [x] x,y coordinates
    - [x] map height and width
    - [x] number of Access Points `AP` and Stations `STA`
    - [x] minimum and maximum distance between one AP and the other
    - [x] minimum and maximum distance between one AP and the Stations associated with it
    - [x] channel bonding
    - [x] central frequency
    - [x] and much more [refer here](https://github.com/bhu1-103/cappy-definitive-edition/blob/main/step1/input-java.csv)

## [STEP 2](https://github.com/bhu1-103/cappy-definitive-edition/tree/main/step2)
### Generate the missing Z coordinates.
  - Used [python code](https://github.com/bhu1-103/cappy-definitive-edition/blob/main/step2/oriz.py) to generate z coordinates within the limit/constraint
 - Plot the points
  - [OpenGL program](https://github.com/bhu1-103/cappy-definitive-edition/blob/main/step2/v0.4/source/main.cpp) to plot the APs and STAs in 3D space
  - Used a [modified version](https://github.com/bhu1-103/cappy-definitive-edition/blob/main/step2/v0.4/source/synthwave.cpp) of [my synthwave opengl library](https://github.com/bhu1-103/swiss-army-katana/blob/main/lib/synthwave/synthwave.cpp)
  - [x] full 3D environment
  - [x] traverse along using `WASD` for movement
  - [x] `E` or `Q` for going up or down
  - [x] APs and STAs in differnt sizes and colors for the respective networks
  - [x] room borders
  - [x] OBSS (Overlapping Basic Service Set) displayed

## [STEP 3](https://github.com/bhu1-103/cappy-definitive-edition/blob/main/step3/combine.sh)
### Run Komondor.
 - Used the [pre compiled binary](https://github.com/bhu1-103/cappy-definitive-edition/blob/main/Komondor/Code/main/komondor_main) of [Komondor simulator](https://github.com/wn-upf/Komondor) to find the output parameters.
  - [x] Airtime
  - [x] RSSI (Recieved Signal Strength Indicator)
  - [x] Interference
  - [x] Throughput
  - [x] SINR (Signal to Interference Ratio)
  - [x] Output generated successfully.
  - [ ] Output generated in separate files.

## [STEP 4](https://github.com/bhu1-103/cappy-definitive-edition/blob/main/step4/step4.sh)
### Separate the output parameters.
 - [x] Used a shell script to separate Komondor's single file ouput into separate files and store all ouptuts in their respective folders.


## [STEP 5](https://github.com/bhu1-103/cappy-definitive-edition/blob/main/step5/step5.sh)
