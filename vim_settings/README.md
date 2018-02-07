There are two packages management tools: Pathogen, Vundle. <br />
I prefer to use [Vundle](https://github.com/VundleVim/Vundle.vim)
 
### Prerequisite
#### Install Anaconda with python 3.5
  + Note that VIM is not compatible with ptyhon 3.6+.
    * Latest Anaconda includes python 3.6
  + Thus, install Anaconda 4.2.0. version
  ```sh
  wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh   (for 32-bit)
  wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86.sh      (for 32-bit)
  ```

#### Install latest version of VIM
  + Refer to [instruction](https://github.com/Valloric/YouCompleteMe/wiki/Building-Vim-from-source) in YouCompleMe
  + First, install all the prerequisite libraries, including Git.
  ```sh
  sudo apt install libncurses5-dev libgnome2-dev libgnomeui-dev \
    libgtk2.0-dev libatk1.0-dev libbonoboui2-dev \
    libcairo2-dev libx11-dev libxpm-dev libxt-dev git checkinstall
  ```
  + Second, install vim from source
  ```sh
  mkdir ~/installations
  cd ~/installations
  git clone https://github.com/vim/vim.git
  cd vim
  ./configure --with-features=huge --enable-multibyte --enable-python3interp=yes --with-python3-config-dir=/home/jonghwan/anaconda3/lib/python3.5/config-3.5m/ --enable-gui=gtk2 --enable-cscope --prefix=/usr/local
  make VIMRUNTIMEDIR=/usr/local/share/vim/vim80
  sudo checkinstall
  ``` 
  + Third, set vim as default editor with `update-alternatives`
  ```sh
  sudo update-alternatives --install /usr/bin/editor editor /usr/local/bin/vim 1
  sudo update-alternatives --set editor /usr/local/bin/vim
  sudo update-alternatives --install /usr/bin/vi vi /usr/local/bin/vim 1
  sudo update-alternatives --set vi /usr/local/bin/vim
  ```

### Setting VIM
  + Install [python-vimrc](https://github.com/ets-labs/python-vimrc)
    * Delete existing .vim folder `rm -r ~/.vim`
    * Run `sh -c "$(curl -fsSL https://raw.githubusercontent.com/ets-labs/python-vimrc/master/setup.sh)"`
  + Use vimrc_python_vimrc_revised as ~/.vimrc
  + Install [YouCompleteMe](https://github.com/Valloric/YouCompleteMe)
    * Run `~/.vim/bundle/YouCompleteMe/install.py --clang-completer`
    * If the version of VIM is lower than 7.4.1+, follow the [instruction](https://github.com/Valloric/YouCompleteMe/wiki/Building-Vim-from-source) in YouCompleMe
