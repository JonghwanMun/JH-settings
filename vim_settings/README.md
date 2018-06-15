There are two packages management tools: Pathogen, Vundle. <br />
This vim setting is based on [Vundle](https://github.com/VundleVim/Vundle.vim)
 
### Prerequisite
#### [Deprecated] VIM fails to recognize python in Anaconda, instead use system python.

#### Install latest version of VIM
+ Refer to [instruction](https://github.com/Valloric/YouCompleteMe/wiki/Building-Vim-from-source) in YouCompleMe
+ First, install all the prerequisite libraries, including Git.
    ```sh
    sudo apt-get install libncurses5-dev libgnome2-dev libgnomeui-dev libgtk2.0-dev libatk1.0-dev libbonoboui2-dev libcairo2-dev libx11-dev libxpm-dev libxt-dev python-dev git checkinstall
    ``` 
+ Second, install vim from source using system python 2.7
    ```sh
    mkdir ~/installations
    cd ~/installations
    git clone https://github.com/vim/vim.git
    cd vim
    ./configure --with-features=huge --enable-multibyte --enable-pythoninterp=yes --enable-gui=gtk2 --enable-cscope --prefix=/usr/local --with-python-config-dir=/usr/lib/python2.7/config-x86_64-linux-gnu
    make VIMRUNTIMEDIR=/usr/local/share/vim/vim81
    sudo checkinstall
    ```    
+ Third, set vim as default editor with `update-alternatives`
    ```sh
    sudo update-alternatives --install /usr/bin/editor editor /usr/local/bin/vim 1
    sudo update-alternatives --set editor /usr/local/bin/vim
    sudo update-alternatives --install /usr/bin/vi vi /usr/local/bin/vim 1
    sudo update-alternatives --set vi /usr/local/bin/vim
    ```

### Setting VIM environment (easy setting)
+ Install [python-vimrc](https://github.com/ets-labs/python-vimrc)
    * Delete existing .vim folder `rm -r ~/.vim`
    * Run `sh -c "$(curl -fsSL https://raw.githubusercontent.com/ets-labs/python-vimrc/master/setup.sh)"`
+ Use vimrc_python_vimrc_revised as ~/.vimrc
+ Install [YouCompleteMe](https://github.com/Valloric/YouCompleteMe)
    * Run `~/.vim/bundle/YouCompleteMe/install.py --clang-completer`
    * If the version of VIM is lower than 7.4.1+, follow the [instruction](https://github.com/Valloric/YouCompleteMe/wiki/Building-Vim-from-source) in YouCompleMe

### Check python version to compile VIM
+ In vim ` :python -c "import sys; print(sys.version)" `
