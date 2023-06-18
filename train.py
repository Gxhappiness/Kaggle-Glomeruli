# -*- coding = utf-8 -*-
# @Time : 2023/6/18 12:27
# @Author : Happiness
# @File : train.py
# @Software : PyCharm



####载入config配置文件

from mmengine import Config
cfg = Config.fromfile('Glomeruli_pspnet_cityscapes.py')



####准备训练

from mmengine.runner import Runner
from mmseg.utils import register_all_modules

# register all modules in mmseg into the registries
# do not init the default scope here because it will be init in the runner
register_all_modules(init_default_scope=False)
runner = Runner.from_cfg(cfg)



###开始训练

if __name__=='__main__':#不加这个会报错，多线程错误
    runner.train()