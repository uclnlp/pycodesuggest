# Learning Python Code Suggestion with a Sparse Pointer Network

This repository contains the code used in the paper "Learning Python Code Suggestion with a Sparse Pointer Network"

## Prerequisites
* [Python 3.5+](https://www.python.org/)
* [Git 1.7+](https://git-scm.com/)
* [Tensorflow 0.9+](www.tensorflow.org) (Tested on 0.9 - 0.11)
* [Github3.py](https://github3py.readthedocs.io/en/master/)
* [GitPython](http://gitpython.readthedocs.io/en/stable/intro.html)
* An account at [Github](https://github.com)

## Generating the Corpus
### Step 1: Cloning the Repos
To recreate the corpus used in the paper, run:

`python3 github-scraper/scraper.py --mode=recreate --outdir=<PATH-TO-OUTPUT-DIR> --dbfile=data/cloned_repos.dat --githubuser=<GITHUB USERNAME>`

Where outdir is the path on your local machine where the repos will be cloned. You will be prompted for your Github password.

---

To obtain a fresh corpus based on a new search of Github, using the same criteria as the paper, run:

`python3 github-scraper/scraper.py --mode=new --outdir=<PATH-TO-OUTPUT-DIR> --dbfile=cloned_repos.dat --githubuser=<GITHUB USERNAME>`

Note that you may interrupt the process and continue where it left off later by providing the same dbfile. 

---

There are a number of other parameters that allow you to create your own custom corpus, specifying the *programming language* or search term used to query Github amongst others. Run  `python3 github-scraper/scraper.py -h` for more information

### Step 2: (OPTIONAL): Remove unnecessary files
Linux/Mac OS: Run the following command in your output directory to remove non Python files

`find . -type f ! -name "*.py" -delete`

### Step 3: Normalisation
Run the following command to normalise all files with a .py extension by providing the output directory of step 1 as the path. The normalised files will be written to a new directory with "normalised" appended to the path. 

`python3 github-scraper/normalisation.py --path=<PATH TO DOWNLOADED CORPUS>`

Files which can't be parsed as valid Python3 will be ignored. The list of successfully processed files is written to PATH/processed.txt which also allows for the normalisation to continue if interrupted. 

### Step 4: Split into train/dev/test
To use the same train/dev/test split as used in the paper, copy the files train_files.txt, valid_files.txt and test_files.txt from the data directory into the downloaded corpus and normalised corpus directories.

---

To generate a new split, run the following command which generates the list of train files (train_files.txt), validation files (valid_files.txt) and test files (test_files.txt) in the ratio 0.5/0.2/0.3. Use the **normalised** path from the previous step. This will ensure that the list of files is available in both the normalised and unnormalised data sets. 

`python3 github-scraper/processFiles.py --path=<PATH TO NORMALISED CORPUS>`

Then copy the 3 generated lists to the original un-normalised path. 


