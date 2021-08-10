# install vim v8.2
sh vim_v8.2_install.sh

# install vimrc
rm -r ~/.vim
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ets-labs/python-vimrc/master/setup.sh)"
cp vimrc_python_vimrc_revised ~/.vimrc

# install c++-8
sudo apt install gcc-8 g++-8
#ls -la /usr/bin/ | grep -oP "[\S]*(gcc|g\+\+)(-[a-z]+)*[\s]" | xargs bash -c 'for link in ${@:1}; do echo sudo ln -s -f "/usr/bin/${link}-${0}" "/usr/bin/${link}"; done' 8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# install cmake > 1.4
sh cmake_install.sh

# install youcompleteme
cd ~/.vim/bundle/YouCompleteMe
#git checkout d98f896
~/.vim/bundle/YouCompleteMe/install.py --clang-completer
