# Juerka

## Overview
### What is this?
This is the code intended to simulate spiking neural network activities.
I was impressed by Mr.Izhikevich's great simulation (https://www.izhikevich.org/publications/spikes.htm) and I try to create "yet another" spiking neural network simulation somehow optimized for x86_64 architecture.

### The purpose of this?
This project is intended to propose the basis for a large scale spiking neural network simulation.
Experiments, something interesting, will be performed in fork or derived projects.
And some utilities, what I currently imagine is the visualization tools, will also be prepared in another projects. -> ![Juerka-Utility-Collection](https://github.com/Junichi-Juerka-Suzuki/Juerka-Utility-Collection)

### Who contributes to this?
Myself.

### About sponsors
If you like it, please donate!

Thank you!.
ありがとうございます。

## Technical details
### How to run the simulation?
#### Windows environment
##### Prerequisits
* Microsoft Visual Studio 2022 Version 17.10.0 Preview 2.0 or later is recommended.
  - CMake package
  - Linux cross build package
* WSL2(Ubuntu) environment for cross building.

##### build & run
1. Clone the repository and open it with Microsoft Visual Studio 2022.
1. And then set the build target as the WSL: Ubuntu.
1. Set debug target as the Juerka and run.
1. Then you can find \"log/YYYYmmDDHHMMSS\" folder as in the same folder as the Juerka executable file.

I will write down more later.

#### Linux environment

I will write down more later.

### How to visualize the simulation result?
#### Prerequisits
* Microsoft Visual Studio 2022 Version 17.10.0 Preview 2.0 or later is recommended.
  - Python build package
* Python 3.9.2 or later is recommended.
* Numpy 1.26.4 or later is recommended.
* matplotlib 3.8.3 or later is recommended.
* [Juerka-Utility-Collection](https://github.com/Junichi-Juerka-Suzuki/Juerka-Utility-Collection)

#### raster plot
##### Windows environment

I will explain later.

##### Linux environment
1. copy the path of \"YYYYmmDDHHMMSS\" folder mentioned [here](https://github.com/Junichi-Juerka-Suzuki/Juerka?tab=readme-ov-file#build--run).
2. pass the folder path as the \"-dirname\" argument of [Juerka-Utility-Collection](https://github.com/Junichi-Juerka-Suzuki/Juerka-Utility-Collection#linux-environment).
3. Choose which log file you would like to visualize and then pass the index of it as the \"-logfile_index\" argument of [Juerka-Utility-Collection](https://github.com/Junichi-Juerka-Suzuki/Juerka-Utility-Collection#linux-environment).
