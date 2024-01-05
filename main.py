import argparse
import sys
import os
import shutil
import zipfile
import time

# torchlight
import torchlight
from torchlight import import_class

from processor.processor import init_seed
init_seed(0)

def save_src(target_path):
    code_root = os.getcwd()
    srczip = zipfile.ZipFile('./src.zip', 'w')
    for root, dirnames, filenames in os.walk(code_root):
            for filename in filenames:
                if filename.split('\n')[0].split('.')[-1] == 'py':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
                if filename.split('\n')[0].split('.')[-1] == 'yaml':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
                if filename.split('\n')[0].split('.')[-1] == 'ipynb':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
    srczip.close()
    save_path = os.path.join(target_path, 'src_%s.zip' % time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))
    shutil.copy('./src.zip', save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')
    processors = dict()

    processors['pretrain_aimclr'] = import_class('processor.pretrain_aimclr.AimCLR_Processor')
    processors['pretrain_mbaimclr'] = import_class('processor.pretrain_mbaimclr.MBAimCLR_Processor') #added
    processors['pretrain_transclr'] = import_class('processor.pretrain_transclr.TransCLR_Processor') #added
    processors['linear_evaluation'] = import_class('processor.linear_evaluation.LE_Processor')
    processors['linear_evaluation_mb'] = import_class('processor.linear_evaluation_mb.MBLE_Processor') #added

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()
    
    #added
    # Loop that will evaluate the model from every 5 epochs of training (training it for 1 epoch each time)
    if arg.loop:
        for i in range(5,arg.lepch + 5,5):

            if arg.processor == 'linear_evaluation':
                arglist = sys.argv[2:]+ ['--weights', 
                                        f'/home/hinard/MBAimCLR/data/gty/AAAI_github/ntu60_cv/aimclr_joint/pretext/epoch{i}_model.pt',
                                        '--mod_epoch', f'{i}']
                
            elif arg.processor == 'linear_evaluation_mb':
                arglist = sys.argv[2:]+ ['--weights', 
                                        f'/home/hinard/MBAimCLR/data/gty/AAAI_github/ntu60_cv/transclr_joint/pretext/epoch{i}_model.pt',
                                        '--mod_epoch', f'{i}']
            else:
                raise ValueError
            
            Processor = processors[arg.processor]
            p = Processor(arglist)
            p.start()
            del p

    else:
        # start
        Processor = processors[arg.processor]
        p = Processor(sys.argv[2:])
        p.start()

    if p.arg.phase == 'train':
        # save src
        save_src(p.arg.work_dir)

    
