command example:
    python main.py --conf=test.ini --id=1  --model=ARSE

ARSE
python main.py --conf=xxxx/conf/epinions/arse.ini --id=1 --model=ARSE

Usage:
Step1:
Replace the xxxx in src/main.py and /conf/epinions/arse.ini with your own code path.

Step2:
Cd src, execute python main.py --conf=xxxx/conf/epinions/arse.ini --id=1 --model=ARSE, the script will output files to root_dir, and you can configure the root_dir in arse.ini

Environment:
1.python2.7
2.tensorflow_1.8.0 / 1.11.0, cpu or gpu is ok
3.numpy, ipdb, termcolor, ConfigParser, no version limited