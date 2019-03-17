## HOW TO RUN THE PROGRAM

1. Move to the folder with file **_run.py_** via Terminal, i.e.:

    ````
    cd /path/to/jetsta
    ````
2. Type command in Terminal

    ``````
    python run.py
    ``````

It is necessary to have Python 3.5 or newer to run the program.
To make sure, type command in Terminal:

````
python -V
````

If displayed version is less than 3.5, it is posible to switch version by typing following commands:

````
module unload python
module load python
````

These commands will switch active version of python to the latest one.

## OUTPUT

Output variables:

1. Inversion radius of crash
2. Crash duration

JETSTA return results in several ways:

1. Print all results in terminal during computing
2. Exprort all results into file **_results/output.csv_**, which is excel-like file.
3. Create images in **_results_** folder to clarify results.

## ABOUT PARAMETERS
#### SETTING PARAMETERS

In the same directory where **_run.py_** is placed, you can find file **_input.json_**

That file can be opened with any text editor. Inside is placed structured list of parameters which should be filled.

#### REQUIRED PARAMETERS

Obligatory parameters are **_required.crashes.start_** and **_required.crashes.end_**.

Both are integer numbers - order of crash in MatLab database.

If **_required.crashes.start_** and **_required.crashes.end_** are different, _i.e. 1 and 10_, the program will analyse every crash from 1 to 10.

If **_required.crashes.start_** and **_required.crashes.end_** are the same number, _i.e. 1 and 1_, the program will analyse only crash number 1.

#### ADVANCED PARAMETERS (IN DEV)

**_advanced.median_filter_window_size_** - 

**_advanced.close_plots_** - 

**_advanced.time_smooth_for_rinv_** - 

**_advanced.window_filtration_function_** - 

**_advanced.channels.start_** - 

**_advanced.channels.end_** - 

## CONTACT
On any question regarding JETSTA, please contact Andrii Tishchenko, tishchenko.andrii@gmail.com

GitHub: https://github.com/azarat/jetsta/
