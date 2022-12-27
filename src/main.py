import os
from run import run
from utils.param_utils import print_parameters
from multiprocessing import Process

COMPARISON = True
# COMPARISON = False


def main():
    from parameters import parameters
    params = parameters.Parameters()
    print_parameters(params, params.param_overlapped_dict)

    for run_number in range(params.RUNS):
        run(params, run_number)
        print("Run:{}/{} is finished".format(run_number, params.RUNS))

    print("Exiting script")

    os._exit(os.EX_OK)


def comparison_main():
    from parameters import multi_parameters
    params_list = multi_parameters.MultiParameters().params_list
    RUNS = params_list[0].RUNS

    for run_number in range(RUNS):
        ps = [Process(target=run, args=(params, run_number)) for params in params_list]
        for p in ps:
            p.daemon = False
            p.start()
        for p in ps:
            p.join()

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


if __name__ == '__main__':
    if COMPARISON:
        comparison_main()
    else:
        main()
