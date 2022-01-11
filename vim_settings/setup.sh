# install vim v8.2
sh vim_v8.2_install.sh

# install cmake > 1.4
sh cmake_install.sh

# install vimrc
rm -r ~/.vim
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ets-labs/python-vimrc/master/setup.sh)"
cp vimrc_python_vimrc_revised ~/.vimrc

# install neocomplete
git clone https://github.com/Shougo/neocomplete.vim.git ~/.vim/pack/vendor/start/neocompete.vim
