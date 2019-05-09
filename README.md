# 重要事项说明

已完成的工作、正在进行中的工作以及将要做的工作可以查看Projects（还没弄好）。在本README中会有一个总结。

## 重点！
* 通过运行run_picar.py来运行程序，终止程序时务必使用`ctrl C`退出，否则捕获Camera的进程在退出程序后仍被占用。

树莓派上的工程环境是
* 架构：armv7l
* 系统：Raspbian（基于Linux）
* Python版本：3.5.x
* OpenCV版本：3.4.2.16

在自己电脑上进行编程时注意代码是否可移植到树莓派上，因为环境不同可能出现的问题有
1. 由于系统架构的不同所以一些程序中依赖的库不存在或不存在相同的版本。因为树莓派采用armv7l架构，所以与之匹配的apt软件、PyPi库相对较少，所以在使用新的package前最好检查其是否有对应的armv7l版本，且与Python3.5.x适配。
2.  由于OpenCV版本的不同所以一些函数的用法可能不同。

树莓派的连接方式：
* 用*SSH*进行连接（终端方式）
* 用*VNC Viewer*进行连接（图形化界面）
* 用*Windows远程桌面连接*进行连接（图像化界面）（推荐）

注意：
1. 希望大家规范书写代码，按照[Google开源项目风格指南（Python）](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/contents/)书写代码。主要要关注的点是**注释**、**命名**规范。
2. 新版代码及时上传到仓库里，上传前检查代码是否完整规范，删除掉无用的代码。
3. 建议采用PyLint和PEP8的linter。
