# Paper
[Develop A Navigation Approach for Mobile Robots Based on the Distributional Deep Reinforcement Learning Framework](https://ieeexplore.ieee.org/abstract/document/10582923)

# Simulation Environment
```bash
https://github.com/sontypo/Differential_Robot_Gazebo_Simulation
```
# Build DRL image
```bash
docker build -t deep_reinforcement_learning .
```
# Run DRL image
```bash
docker run -it --rm \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    deep_reinforcement_learning
```