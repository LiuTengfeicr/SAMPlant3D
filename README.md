There is a method for splitting point clouds using SAM（Revisions）
SAMPlant3D (or SAMPoint) for single-plant segmentation of population plants

## Dataset

For readers' convenience, we provide a reduced version of the dataset, i.e., 50 point cloud data: download 
## Setting

See the SAM project for specific environment configurations: [SAM](https://github.com/facebookresearch/segment-anything.git)

The missing packages are configured according to the current environment

## Start

Download [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), create the file and place it under [SAM/model](./SAM/model)

Create the file output, and its sub-files measure, pointcloud, sammask, sampic

The format of the completed file is as follows
```
SAMPlant3D/
│
├── data/
│   ├── 1_1_0-3-29.txt
│   ├── 2_1_0-3-30-2.txt
│   └── ...
│
├── output/
│   ├── measure
│   ├── pointcloud
│   ├── sammask
│   └── sampic
│ 
├── pic_check
├── SAM/
│   ├── model/
│   │   └── sam_vit_h_4b8939.pth
│   ├── scripts
│   └── segment_anything

├── utils
├── main.py
├── start.py
```
## Run

Run the program by typing ```python start.py --file_path data ``` in the terminal

Main Parameters
```
--output_pointcloud <Save address of the point cloud after segmentation>
--output_measure <Path to output phenotype parameters>
--alpha_v --alpha_v <Parameterization of alpha volume and area>
--method_noise <Denoising method selection: median/gaussian/bilateral>
--method_Volume <Volume Calculation Options: Alpha/Normalized>
--measure <Whether to choose to calculate phenotype parameters, default is False>
```
## Results
```measure``` saves the results of the run, including phenotypic parameters, iou, pre and other metrics, ```pointcloud``` output saves the segmented point cloud, ```sammask``` and ```sampic``` save the path to the masked image for the output.

## Additional notes
```pic_check``` file for the picture screening algorithm, run ```picture_check.py``` to calculate the clarity of the picture, run ```picture_check_test.py``` to draw a picture clarity line graph, so that you can screen blurred pictures, of which the three ```json``` files for the results of the preservation of the example
