### Setting tmux environment (easy setting)
+ Install [gpakosz/.tmux](https://github.com/gpakosz/.tmux.git)
``` bash
cd
git clone https://github.com/gpakosz/.tmux.git
ln -s -f .tmux/.tmux.conf
cp .tmux/.tmux.conf.local .
```
+ Open .tmux.local and uncomment following lines
	* set -g mouse on (line 279)
	* only use C-a instead both using C-a & C-b (line 287-291)
+ Open .tmux.conf and revise following lines
	* set -sg repeat-time 600 -> set -sg repeat-time 0 (line 15) ==> prevent repeating operations
