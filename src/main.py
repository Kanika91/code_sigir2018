import sys, getopt
import ConfigParser as cp

sys.path.append('/home/knarang2/code_sigir2018/src/class')
sys.path.append('/home/knarang2/code_sigir2018/src/public')
sys.path.append('/home/knarang2/code_sigir2018/src/arse')

def usage():
    print(u"""
        -h: usage help
        -c,--conf: configure path
        -m,--model: which model
        --id: experiment record id
    """)

def execute_model(conf, model, record_id):
    if model == 'ARSE':
        import arse
        arse.start_model(conf, record_id)

opts, args = getopt.getopt(sys.argv[1:], "hD:c:m:", ["conf=","model=","id=","eva="])

for op, value in opts:
    if op == "-h":
        usage()
        sys.exit()
    elif op == "--conf" or op == "-c":
        print value
        config = value
    elif op == '--eva':
        eva_model(config, value)
    elif op == "--id":
        print value
        record_id = value
    elif op == "--model":
        print value
        execute_model(config, value, record_id)
