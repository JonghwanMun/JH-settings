There are two packages management tools: Pathogen, Vundle. <br />
I prefer to use [Vundle](https://github.com/VundleVim/Vundle.vim)

### Install Vundle
Vundle is a convenient package manager for VIM. (You do not need to install Vundle.)
  ```
  git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
  ```

### Easy Setting for Python, C, C++
  + Install [python-vimrc](https://github.com/ets-labs/python-vimrc)
    * Delete existing .vim folder `rm -r ~/.vim`
    * Run `sh -c "$(curl -fsSL https://raw.githubusercontent.com/ets-labs/python-vimrc/master/setup.sh)"`
  + Use vimrc_python_vimrc_revised as ~/.vimrc
  + Install [YouCompleteMe](https://github.com/Valloric/YouCompleteMe)
    * Run `~/.vim/bundle/YouCompleteMe/install.py --clang-completer`
    * If the version of VIM is lower than 7.4.1+, follow the [instruction](https://github.com/Valloric/YouCompleteMe/wiki/Building-Vim-from-source) in YouCompleMe
    * To install VIM 8.0, use following configuration (note revise with-python3-config-dir path)
    ```sh
    cd ~
    git clone https://github.com/vim/vim.git
    cd vim
    ./configure --with-features=huge --enable-multibyte --enable-rubyinterp=yes --enable-python3interp=yes --with-python3-config-dir=/home/jonghwan/anaconda3/lib/python3.6/config-3.6m-x86_64-linux-gnu --enable-gui=gtk2 --enable-cscope --prefix=/usr/local
    make VIMRUNTIMEDIR=/usr/local/share/vim/vim80
    ```

### List of Useful Flugins
  + `scrooloose/nerdtree`: Project and file navigation
  + `majutsushi/tagbar`: Class/module brower
