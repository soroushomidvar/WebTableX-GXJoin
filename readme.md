# GXJoin
The resources in this repository were developed for the following paper:
> S. Omidvartehrani, A. Dargahi Nobari, and D. Rafiei. “GXJoin: On the Generalization of Transforming Tables for Explainable Joinability”

For more details, please refer to the [paper](TBA).

## Usage
The main source files located in the `src` directory:

+ `main.py`: This file is the main to run our approach, CST, and the implementation of auto-join on a set of tables. It can be called with no command-line argument to use default values for all parameters.

    To set the method, parameters, and paths, a config json file may be passed as a command-line argument:

    > python3 src/main.py -c config.json

    Multiple sample config files are provided in the `config` directory. 

+ `transformation_joiner.py`: This file can parse the transformations generated by `main.py` and utilize them for table join.

+ `join.py`: A simple example of an end to end join with our approach. For advanced settings and a detailed output on transformations and join process, `main.py` and `transformation_joiner.py` should be used.



## Citation

If you have used the codes in this repository, kindly cite the following paper:

> TBA