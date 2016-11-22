from github3 import login, GitHub
from getpass import getpass
import pickle
import os
from PyRepo import PyRepo
import time
import getopt
import sys


def login_github(githubUser):
    password = getpass('GitHub password for {0}: '.format(githubUser))
    g = login(githubUser, password)
    return g


def new(repos, githubUser, searchQuery, min_stars, language, limit, outputDirectory, dbFile):
    g = login_github(githubUser)
    query = searchQuery + (" " if len(searchQuery) > 0 else "") + "stars:>" + str(min_stars) + " language:" + language

    searchResults = g.search_repositories(query, number=limit, sort="forks")

    for result in searchResults:
        name = result.repository.name
        full_name = result.repository.full_name
        description = result.repository.description
        clone_url = result.repository.clone_url
        num_stars = result.repository.watchers
        num_forks = result.repository.forks_count
        created_at = result.repository.created_at
        pushed_at = result.repository.pushed_at

        repo = PyRepo(name, full_name, description, clone_url, time.time(), num_stars, num_forks, created_at, pushed_at)
        if repo in repos:
            print("Skipping %s because it has already been cloned" % repo)
        else:
            try:
                repo.clone(outputDirectory)
                repos.append(repo)
                print("Cloned %s" % repo.details())

                outfile = open(dbFile, "wb")
                pickle.dump(repos, outfile)
                outfile.close()

            except Exception as e:
                print("Failed to clone %s due to %s" % (repo, e))


def recreate(repos, outputDirectory):
    for repo in repos:
        repo.checkout(outputDirectory)
        print("Checked out: %s" % repo.details())


def create_repos(dbFile):
    repos = []
    if os.path.exists(dbFile):
        infile = open(dbFile)
        repos = pickle.load(infile)
        infile.close()
    return repos


def printusage():
    print("Usage:\n\tscraper.py [parameters]\n")

    print("Required Parameters:")
    print("\t-m --mode\t\t\t'new' to generate new corpus or 'recreate' to clone corpus from dbfile")
    print("\t-o --outdir\t\t\tDirectory into which repos are cloned")
    print("\t-d --dbfile\t\t\tList of repos to clone in 'recreate' mode or save in 'new' mode")
    print("\t-u --githubuser\t\tGithub username")

    print()
    print("Optional Parameters:\n")
    print("\t-h --help\t\t\tShow this help")
    print("\t-n --limit\t\t\tNumber of repos to obtain in 'new' mode. DEFAULT 1000")
    print("\t-s --search\t\t\tSearch query used in 'new' mode. DEFAULT '<blank>'")
    print("\t-t --stars\t\t\tMinimum number of stars threshold. Used in 'new' mode to search. DEFAULT 100")
    print("\t-l --language\t\tRepo language, used in 'new' mode to search. DEFAULT 'python'")


def main(argv):
    mode = ""
    outputDirectory = ""
    dbFile = ""
    githubUser = ""

    limit = 1000
    searchQuery = ""
    min_stars = 100
    language = "python"

    try:
        opts, args = getopt.getopt(argv, "hlstnm:o:d:u:",
                                   ["mode=", "outdir=", "dbfile=", "githubuser=", "limit=", "search=",
                                    "stars=", "language=", "help"])
    except getopt.GetoptError:
        printusage()
        raise

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            printusage()
            sys.exit()
        elif opt in ("-m", "--mode"):
            mode = arg
        elif opt in ("-o", "--outdir"):
            outputDirectory = arg
        elif opt in ("-d", "--dbfile"):
            dbFile = arg
        elif opt in ("-u", "--githubuser"):
            githubUser = arg
        elif opt in ("-n", "--limit"):
            limit = int(arg)
        elif opt in ("-s", "--search"):
            searchQuery = arg
        elif opt in ("-t", "--stars"):
            min_stars = int(arg)
        elif opt in ("-l", "--language"):
            language = arg

    if mode == '' or outputDirectory == '' or dbFile == '' or githubUser == '':
        printusage()
        sys.exit(2)

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    os.chdir(outputDirectory)
    repos = create_repos(dbFile)

    if mode == "new":
        new(repos, githubUser, searchQuery, min_stars, language, limit, outputDirectory, dbFile)
    elif mode == "recreate":
        recreate(repos, outputDirectory)
    else:
        print("Mode parameter must be 'new' or 'recreate'")

    print("Done")


if __name__ == "__main__":
    main(sys.argv[1:])







