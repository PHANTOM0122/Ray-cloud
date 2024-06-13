# Ray-cloud 

## [CVPR 2024] Efficient Privacy-Preserving Visual Localization Using 3D Ray Clouds
**Authors:** [Heejoon Moon](https://github.com/PHANTOM0122), [Chunghwan Lee](https://github.com/Fusroda-h), [Je Hyeong Hong](https://sites.google.com/view/hyvision)

*************************************
### :rocket: **News** 
:fire: [2024.06.13] We're now working on releasing our code and please keep stay tuned for the final release!

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
We borrowed the implementation of localization and inversion framework from PPL repository. </br>
Thanks to [Chunghwan Lee](https://github.com/Fusroda-h) for your contribution. </br>

## Citation
```bibtex
@InProceedings{Moon_2024_CVPR,
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

