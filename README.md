# DLAV_3DHumanPose



This work is mainly done based on the following github repo https://github.com/redrock303/HEMlets. The folder HEMlets contains their code of the repo.



Following are the two papers linked to the HEMlets code:
HEMlets Pose: [Learning Part-Centric Heatmap Triplets for Accurate 3D Human Pose Estimation](https://arxiv.org/pdf/1910.12032.pdf)

[HEMlets PoSh: Learning Part-Centric Heatmap Triplets for 3D Human Pose and Shape Estimation](https://arxiv.org/pdf/2003.04894.pdf)


To run our code several parts need to be downloaded: 

	- The pre-trained weights: donwnload 'ckpt' (hemlets_h36m_lastest.pth) and 'data' (S_11_C_4_1.h5)   (https://drive.google.com/drive/folders/1z8Jj0xx4SvHC-YKuw_M_c_Z4vA4HpzID?usp=sharing)

	- Download the yolo weights (classes.txt, yolov3.weights and yolov3.cfg)

	- Download the Human3.6M dataset (SCITAS - h3.6)

To test on the videos from the class (test4_cut.mp4 for example), download the videos from [Google Drive](https://drive.google.com/drive/folders/16xf0AF9zgWAuK6Xyr5xK85t77hM3BwAv?usp=sharing)



The following architechture must be followed:


- ckpt
	- classes.txt
	- yolov3.weights
	- yolov3.cfg
	- hemlets_h36m_lastest.pth

- data
	- S11
		- S_11_C_4_1.h5
	- test_set
		- test4_cut.mp4
	- data_2d_h36m.npz
	- data_3d_h36m.npz
	- Directions.55011271.mp4 (and other videos from the Human3.6M)

- DLAV
	- checkpoints
	- codes

- HEMlets
	- requirement.txt

- Inference_in_video


Create a new Conda environement:
```conda create -n DLAV python=3.7.7```

Install the needed librairies 
```pip install -r requirement.txt```

To run the training code. go to the DALV folder and either run 
```bash train.sh```
or 
```python train.py --dataset_path '../data/dataset.h5' --epochs 30 --batch_size 4 --num_workers 1 --lr 0.0001 --save_interval 1 --checkpoint_dir 'checkpoints' --data_ratio 0.4 --validation_ratio 0.1 --test_ratio 0.0```


To run the RealTime go to the RealTime folder and run ```bash inference.sh```

To run the tests on the video from the class go to the Inference_on_video folder and run ```bash inference.sh```







This work is mainly done based on the following github repo https://github.com/redrock303/HEMlets. The folder HEMlets contains their code of the repo.



And following the two linked papers:
HEMlets Pose: Learning Part-Centric Heatmap Triplets for Accurate 3D Human Pose Estimation (https://arxiv.org/pdf/1910.12032.pdf)

HEMlets PoSh: Learning Part-Centric Heatmap Triplets for 3D Human Pose and Shape Estimation (https://arxiv.org/pdf/2003.04894.pdf)



Copied and pasted from the original git (mentionned above)
"
We provide quick inference code to validate and visualize our results on [Human3.6M](http://vision.imar.ro/human3.6m/description.php). Brief runing instructions are given below.
1. Pre installation\
 create a new conda vitual environment\
 conda/pip install -r requirement.txt
2. Download the pre-trained model and the tiny dataset(a pre-processed testing video from Human3.6M) at [Baidu Cloud](https://pan.baidu.com/s/1pg35KvvqUK5jX8UMRk_emQ) [code:HEMs] or [Google Cloud](https://drive.google.com/drive/folders/1z8Jj0xx4SvHC-YKuw_M_c_Z4vA4HpzID?usp=sharing).
3. Visualization and evaluation on a single video from Human3.6M.
   We implement a script for visualization and evaluation of the predicted results on Human3.6M by running the command:\
   ```bash inference.sh```\"
