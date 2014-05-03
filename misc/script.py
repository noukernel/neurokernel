import os
import sys
from subprocess import call

path = os.getcwd()
c_path = path.split('/')
new_path = '/'
for ii in c_path[1:-1]:
	new_path += ii + '/'

print new_path

os.chdir(new_path)

call([sys.executable, 'setup.py', 'install'])

os.chdir(path)

call([sys.executable, 'neurotest.py'])
