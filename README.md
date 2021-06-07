
# Deep Grasping ROS

- ROS wrapper for DNN based robotic grasping algorithms
- Support 6-DoF-GraspNet [[paper]](https://arxiv.org/abs/1905.10520) [[code]](https://github.com/NVlabs/6dof-graspnet)
- Support Contact-GraspNet [[paper]](https://arxiv.org/abs/2103.14127) [[code]](https://github.com/NVlabs/contact_graspnet)

## TO DO
- support GGCNN
- support GQCNN
- publish the grasping point as TF
- add install documentation


## 6-DoF-GraspNet

<img src="./imgs/6dof_grasp.png" height="250">


### Setup

```
conda create -n 6dofgraspnet python=2
python -m pip install -r requirements.txt
cd src && git clone https://github.com/SeungBack/6dof-graspnet
conda activate 6dofgraspnet && pip install -r requirements.txt
```

### RUN

Azure kinect node
```
ros27 && ROS_NAMESPACE=azure1 roslaunch azure_kinect_ros_driver driver.launch color_resolution:=1536P depth_mode:=WFOV_UNBINNED fps:=5 tf_prefix:=azure1_
```

6-dof-graspnet server
```
ros && conda activate 6dofgraspnet && cd ~/catkin_ws/src/deep-grasping-ros/src \
    && python 6dgn_ros_server.py
```

6-dof-graspnet client
```
ros && conda activate 6dofgraspnet && \
    cd ~/catkin_ws/src/deep-grasping-ros/src/6dof-graspnet \
    && python -m demo.6dgn_client --vae_checkpoint_folder checkpoints/npoints_1024_train_evaluator_0_allowed_categories__ngpus_1_/
```




## Contact-GraspNet

<img src="./imgs/contact_grasp.png" height="250">

### RUN

Azure kinect node
```
ros27 && ROS_NAMESPACE=azure1 roslaunch azure_kinect_ros_driver driver.launch color_resolution:=1536P depth_mode:=WFOV_UNBINNED fps:=5 tf_prefix:=azure1_
```

contact graspnet server
```
ros && conda activate contact_graspnet_env \
    && cd ~/catkin_ws/src/deep-grasping-ros/src \
    && python contact_grasp_server.py
```

contact graspnet client

```
ros && conda activate contact_graspnet_env \
    && cd ~/catkin_ws/src/deep-grasping-ros/src/contact_graspnet \
    && CUDA_VISIBLE_DEVICES=1 python contact_graspnet/contact_grasp_client.py
```






## Authors
* **Seunghyeok Back** [seungback](https://github.com/SeungBack)