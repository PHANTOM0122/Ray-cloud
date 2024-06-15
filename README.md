# Ray-cloud 

## [CVPR 2024] Efficient Privacy-Preserving Visual Localization Using 3D Ray Clouds
**Authors:** [Heejoon Moon](https://github.com/PHANTOM0122), [Chunghwan Lee](https://github.com/Fusroda-h), [Je Hyeong Hong](https://sites.google.com/view/hyvision)

*************************************
### :rocket: **News** 
:fire: [2024.06.15] We're releasing our intitial code and now working on cleaning our code. Please keep stay tuned for the final release!

*************************************
:page_with_curl: \[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Moon_Efficient_Privacy-Preserving_Visual_Localization_Using_3D_Ray_Clouds_CVPR_2024_paper.pdf)] \[[Supplementary document](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Moon_Efficient_Privacy-Preserving_Visual_CVPR_2024_supplemental.pdf)] 
<br/>
**Presentation video:** [![Youtube Badge](https://img.shields.io/badge/Youtube-ff0000?style=flat-square&logo=youtube&link=https://www.youtube.com/channel/UCkWMYftPuCZSBy34Od8KpEw)](https://www.youtube.com/watch?v=oECeygDJ5rY)

**Abstract:** The recent success in revealing scene details from sparse 3D point clouds obtained via structure-from-motion has
raised significant privacy concerns in visual localization.
One prominent approach for mitigating this issue is to lift
3D points to 3D lines thereby reducing the effectiveness of
the scene inversion attacks, but this comes at the cost of increased algorithmic complexity for camera localization due
to weaker geometric constraints induced by line clouds. To
overcome this limitation, we propose a new lifting approach
called “ray cloud”, whereby each lifted 3D line intersects at
one of two predefined locations, depicting omnidirectional
rays from two cameras. This yields two benefits, i) camera localization can now be cast as relative pose estimation between the query image and the calibrated rig of two
perspective cameras which can be efficiently solved using a
variant of the 5-point algorithm, and ii) the ray cloud introduces erroneous estimations for the density-based inversion attack, degrading the quality of scene recovery. Moreover, we explore possible modifications of the inversion attack to better recover scenes from the ray clouds and propose a ray sampling technique to reduce the effectiveness
of the modified attack. Experimental results on two public
datasets show real-time localization speed as well as enhanced privacy-preserving capability over the state-of-theart without overly sacrificing the localization accuracy.
*************************************

## :running: How to run our code!
**Our code built upon the [repository of Paired-Point Lifting (CVPR2023)](https://github.com/Fusroda-h/ppl/tree/main), accessed at June, 2023**. </br>
We borrowed most of the implementation of localization and inversion framework from PPL repository. </br>
Thanks to [Chunghwan Lee](https://github.com/Fusroda-h) for your contribution. </br>

- **Environment setting**

Make a new folder `/Myfolder`.
Make a docker container that fits your environment with a python version 3.9.
Mount the docker volume with the `-v /Myfolder/:/workspace/`.

Clone the git `git clone https://github.com/Fusroda-h/Ray-cloud`
Download `eigen-3.4.0.tar.gz` library from https://eigen.tuxfamily.org/index.php?title=Main_Page to run poselib.

```bash
cd Ray-cloud
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
```

To properly build `poselib`, download the rest of the folders from the [PoseLib](https://github.com/vlarsson/PoseLib).
We only uploaded the customized code from PoseLib implementing P6L solver.

```bash
cd ..
git clone https://github.com/PoseLib/PoseLib.git
# Checkout to the version before refactoring "pybind"
cd PoseLib
git checkout ce7bf181731e4045f990c7e90e93716fe7465d56
# Overwrite customized local poselib to cloned poselib
# And move back to original directory
cd ../
cp -rf Ray-cloud/PoseLib/* PoseLib/
rm -r Ray-cloud/PoseLib
mv PoseLib Ray-cloud/PoseLib
```

Since InvSfM code by Pittaluga et al. is written in tensorflow.v1, Chanhyuk Yun rewritten the whole code to pytorch for the ease of use ([invsfm_torch](https://github.com/ChanhyukYun/invSfM_torch)).
Download pretrained weights from [InvSfM](https://github.com/francescopittaluga/invsfm).
Position the `wts` folder to `utils/invsfm/wts`.
Then, our code will automatically change the weights to torch version and utilize it.

```bash
cd ppl
bash start.sh
```

cf) If you suffer from an initialization error with the message: `avx512fintrin.h:198:11: note: ‘__Y’ was declared here`.

Refer to this [ISSUE](https://github.com/pytorch/pytorch/issues/77939#issue-1242584624) and install with GCC-11

`apt-get install gcc-11 g++-11`

Edit the bash file `start.sh` so that Poselib is compiled with `gcc-11` $-$ substitute `cmake -S . -B _build/ -DPYTHON_PACKAGE=ON -DCMAKE_INSTALL_PREFIX=_install`
to `cmake -S . -B _build/ -DPYTHON_PACKAGE=ON -DCMAKE_INSTALL_PREFIX=_install -DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11`.


If you have other problems in building the packages.
Visit installation each page, s.t. [PoseLib](https://github.com/vlarsson/PoseLib), [Ceres-solver](http://ceres-solver.org/installation.html), [COLMAP](https://colmap.github.io/install.html).
Ubuntu and CUDA version errors might occur.

The codes `database.py` and `read_write_model.py` is from [COLMAP](https://github.com/colmap/colmap).

- **Run the main code (pose estimation, recovering point, restoring image at once)**

You can download example dataset on [Sample_data](https://1drv.ms/u/s!AlaAkmWU9TVG6yIqNBD0PlN43Ewe?e=2gIN1F).
Directories are organized like below.
```bash
├─Dataset_type (energy, cambridge)
│  └─Scene (apt1_living, kingscolledge)
│      ├─bundle_maponly
│      ├─images_maponly
│      ├─query
│      ├─sparse_gt
│      ├─sparse_maponly
│      └─sparse_queryadded
```
The construction of map and queries are explained in [supplementary materials](documents/Lee_et_al_cvpr23_supplemat.pdf).

To generate the PPL-based line cloud and to estimate pose & recover the point cloud from this

```
/usr/local/bin/python main.py
```

You can change your options with the parser in `main.py`.
Or else can manipulate the miute options with `static/variable.py`.

The results are stored in `output` folder.
In the folder, recovered point clouds, pose errors, and recovered image qualities are stored in `Dataset_name/Scene/L2Precon`,`Dataset_name/Scene/PoseAccuracy`,`Dataset_name/Scene/Quality` respectively.
The recovered images will be saved in `dataset/Dataset_name/Scene/invsfmIMG/`.



## Citation
```bibtex
@InProceedings{moon2024raycloud,
    author    = {Moon, Heejoon and Lee, Chunghwan and Hong, Je Hyeong},
    title     = {Efficient Privacy-Preserving Visual Localization Using 3D Ray Clouds},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {9773-9783}
}
```

## License
A patent application for the Raycloud algorithm and the relevant software has been submitted and is under review for registration.
Raycloud is licensed under the CC-BY-NC-SA-4.0 license limiting any commercial use.
[PoseLib](https://github.com/vlarsson/PoseLib) is licensed under the BSD 3-Clause license.

