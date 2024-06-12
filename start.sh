set -e
apt-get update
apt-get upgrade -y
apt-get install cmake gcc g++ vim -y

# Build eigen
echo Build Eigen
if [ -d "eigen-3.4.0/" ]; then
    rm -r eigen-3.4.0/
fi
tar -xf eigen-3.4.0.tar.gz
echo unzip done!
cd eigen-3.4.0
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=// ..
make install
echo Eigen Build done!

pip install pybind11 numpy opencv-python tqdm matplotlib scikit-learn scikit-image torch pandas open3d
apt-get install libgl1-mesa-glx -y

####### Original implmentation
# Build poselib
cd ../../PoseLib
if [ -d "_build" ]; then
    rm -r _build
fi
mkdir _build
cmake -S . -B _build/ -DPYTHON_PACKAGE=ON -DCMAKE_INSTALL_PREFIX=_install
cmake --build _build/ --target install -j 8
cmake --build _build/ --target pip-package
cmake --build _build/ --target install-pip-package
echo "Poselib sucessfully built"


# install colmap
cd ../../
apt-get install -y git cmake build-essential libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev \
                             libboost-system-dev libboost-test-dev libeigen3-dev libsuitesparse-dev libfreeimage-dev \
                             libmetis-dev libgoogle-glog-dev libgflags-dev libglew-dev qtbase5-dev libqt5opengl5-dev \
                             libcgal-dev libpcl-dev
                             
# Ceres solver
if [ -d "ceres-solver" ]; then
    rm -r ceres-solver
fi
if [ -d "ceres-bin" ]; then
    rm -r ceres-bin
fi
git clone -b 2.1.0 https://ceres-solver.googlesource.com/ceres-solver
mkdir ceres-bin && cd ceres-bin
cmake ../ceres-solver
make -j3
make test
make install

# Colmap
cd ../
if [ -d "colmap" ]; then
    rm -r colmap
fi
git clone -b 3.7 https://github.com/colmap/colmap.git
cd colmap
git checkout tags/3.7
mkdir build && cd build
cmake ..
make -j
make install
