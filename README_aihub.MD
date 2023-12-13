# docker run

    cd docker
    docker build -f minimal.Dockerfile -t openpcdet:cuda11 .


# docker run

    xhost local:root && docker run -it --rm -e SDL_VIDEODRIVER=x11 -e DISPLAY=$DISPLAY --env='DISPLAY' --gpus all --ipc host --privileged --network host -p 8080:8081 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /home/ubuntu:/home/ubuntu -v /weights:/weights openpcdet:cuda11 xfce4-terminal --title=openPCDet


# Inner docker env setup

    python3 setup.py develop
    pip install kornia==0.6.5


# Aihub data conversion
1. Download [dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=629)
2. extract at /data/robust/
3. folder structure should be 
    ```
    OpenPCDet
    ├── data
    │   ├── robust
    │   │   │── 1.주간
    ├── pcdet
    ├── tools
    ```

# Aihub data conversion
if python script finds "..../lidar/roof/" directory, copy json and bin files.

    python3 convert.py



# Dataset pre-prosessing

    python3 -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/robust_v1_mini_dataset.yaml


# Train

    cd tools/
    python3 train.py --cfg_file cfgs/custom_models/pointpillar_aihub_robust.yaml --extra_tag "aihub_robust"


# Trained weight visualization

    cd tools/
    python3 demo.py --cfg_file cfgs/custom_models/pointpillar_robust_v1_mini.yaml --ckpt ../output/custom_models/pointpillar_aihub_robust/aihub_robust/ckpt/latest_model.pth --data_path ../data/robust/points
