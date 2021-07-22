cd ~/
wget https://github.com/Kitware/CMake/archive/refs/tags/v3.15.6.zip
unzip v3.15.6.zip
rm v3.15.6.zip
cd CMake-3.15.6
./bootstrap
make -j 16
sudo make install
