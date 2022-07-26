from datetime import datetime
from threading import Timer
# 打印时间函数
import subprocess

command1 = "git add ."
command2 = 'git commit -m "commit" -a'
command3 = "git push origin main"
cmd = "python updata.py"


def printTime(inc):

    print(cmd)
    subprocess.call(cmd, shell=True)
    # print(command1)
    # subprocess.call(command1, shell=True)
    # command2 = 'git commit -m "commit ' + str(count) + '" -a'
    # print(command2)
    # subprocess.call(command2, shell=True)
    # print(command3)
    # subprocess.call(command3, shell=True)
    t = Timer(inc, printTime, (inc,))
    t.start()


# 1d
printTime(86400)
