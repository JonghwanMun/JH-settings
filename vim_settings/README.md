There are two packages management tools: Pathogen, Vundle. <br />
I prefer to use [Vundle](https://github.com/VundleVim/Vundle.vim)

### Install Vundle
Vundle is a convenient package manager for VIM.
  ```
  git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
  ```

### Easy Setting
  + Install [python-vimrc](https://github.com/ets-labs/python-vimrc)
    * Run `sh -c "$(curl -fsSL https://raw.githubusercontent.com/ets-labs/python-vimrc/master/setup.sh)"`
  + Use vimrc_python_vimrc_revised as ~/.vimrc
  + Install [YouCompleteMe](https://github.com/Valloric/YouCompleteMe)
    * Run `~/.vim/bundle/YouCompleteMe/install.py --clang-completer`
    * If the version of VIM is lower than 7.4.1+, follow the [instruction](https://github.com/Valloric/YouCompleteMe/wiki/Building-Vim-from-source) in YouCompleMe

### List of Useful Flugins
  + `scrooloose/nerdtree`: Project and file navigation
  + `majutsushi/tagbar`: Class/module brower
