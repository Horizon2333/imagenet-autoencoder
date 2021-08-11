import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)    
parser.add_argument('--path', type=str)                                                
args = parser.parse_args()

folders = os.listdir(args.path)

if not os.path.exists("list"):
    os.makedirs("list")

fl = open('list/' + args.name + '_list.txt', 'w')
fn = open('list/' + args.name + '_name.txt', 'w')

for i, folder in enumerate(folders):

    fn.write(str(i) + ' ' + folder + '\n')

    folder_path = os.path.join(args.path, folder)
    files = os.listdir(folder_path)

    for file in files:

        fl.write('{} {}\n'.format(os.path.join(folder_path, file), i))

fl.close()
fn.close()
