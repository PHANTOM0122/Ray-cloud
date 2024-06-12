# Ray-cloud 

## [CVPR 2024] Efficient Privacy-Preserving Visual Localization Using 3D Ray Clouds
**Authors:** Heejoon Moon, [Chunghwan Lee](https://github.com/Fusroda-h), [Je Hyeong Hong](https://sites.google.com/view/hyvision)

*************************************
### :rocket: **News** 
:fire: We release the initial version of our implementation. We're now cleaning our code and keep stay tuned for the final release!

*************************************
:grapes: \[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Moon_Efficient_Privacy-Preserving_Visual_Localization_Using_3D_Ray_Clouds_CVPR_2024_paper.pdf)] <br/>
:grapes: \[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Moon_Efficient_Privacy-Preserving_Visual_Localization_Using_3D_Ray_Clouds_CVPR_2024_paper.pdf)] <br/>

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
**Our code is the extension of the [repository of Paired-Point Lifting (CVPR2023)](https://github.com/Fusroda-h/ppl/tree/main), accessed at June,2023**

