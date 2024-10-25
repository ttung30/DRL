FROM osrf/ros:noetic-desktop-full


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y \
    gazebo11 \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    && rm -rf /var/lib/apt/lists/*


ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1


RUN mkdir -p /root/catkin_ws/src

WORKDIR /root/catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"


COPY ./entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh


ENTRYPOINT ["/root/run.sh"]

EXPOSE 11311
EXPOSE 8080
