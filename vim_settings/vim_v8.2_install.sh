# download vim (v8.2) so we can build it from scratch
cd ~
wget https://ftp.nluug.nl/pub/vim/unix/vim-8.2.tar.bz2
tar xf vim-8.2.tar.bz2
cd vim82

# In case Vim was already installed. This can throw an error if not installed, 
# it's the nromal behaviour. That's no need to worry about it
cd src
make distclean

sudo apt install libncurses5-dev libncursesw5-dev libgtk2.0-dev \
	libatk1.0-dev libcairo2-dev libx11-dev libxpm-dev libxt-dev \
	python2-dev python3-dev -y 

export LDFLAGS="-fno-lto"
./configure --with-features=huge \
            --enable-multibyte \
            --enable-rubyinterp=yes \
            --enable-python3interp=yes \
            --with-python3-config-dir=$(python3-config --configdir) \
            --enable-perlinterp=yes \
            --enable-luainterp=yes \
            --enable-gui=gtk2 \
            --enable-cscope \
            --prefix=/usr/local
            
make VIMRUNTIMEDIR=/usr/local/share/vim/vim82
make -j24
sudo make install
