# HomeworkTeX
Homework that are submitted in tex inhabit this repository. Do not disturb!!!


## Requirements
If you like to collaborate on assignments with me but have no knowledge of Git or github, below are some instructions that might be helpful for you.If you have no time, just download the zipped up file of the repository by clicking `Clone or download`.

- [Git](https://git-scm.com/) is a version control system that is used for development of codes/software and other version control tasks. It allows multiple parties to collaborate on a single project/document (in this case the tex file) without ruining each other's version

- GitHub is a web-based Git repository hosting service. It offers all of the distributed version control and source code management (SCM) functionality of Git as well as adding its own features. You can think of Github as a Dropbox for codes instead of general files and folders. You can signup for a github account [here](https://github.com/)

- to start collaborating the Git way, you will need a distribution of Git on your machine.
  - for Windows users I suggest getting the Git for Windows Portable [32bit](https://github.com/git-for-windows/git/releases/download/v2.10.1.windows.1/PortableGit-2.10.1-32-bit.7z.exe)/[64bit](https://github.com/git-for-windows/git/releases/download/v2.10.1.windows.1/PortableGit-2.10.1-64-bit.7z.exe)
  - for Mac users, one can do the installation by following the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)


## How to Git?

Assuming that you have a github account and Git distribution on your machine we are ready to start! Start up your terminal or PortableGit for Mac and Windows system respectively. We will start by cloning this repository. The commands for terminal and PortableGit work the same

**Basics**

|  Command            | Function                                               |
|:-------------------:|--------------------------------------------------------|
| `pwd`               | Returns the present working directory.                 |
| `cd`                | Goes back to the home directory.                       |
| `cd ..`             | Jumps back up one directory.                           |
| `ls`                | List the files in the directory you are in.            |

Tip: `pwd` is used to check which directory you are in, `cd` to change to the directory that you want to access, `ls` is for you to check if the file you are trying to access is in that directory. Use them to navigate around the directories and find your files :)

**Cloning this repository**

```git

git clone https://github.com/zunction/HomeworkTeX *nameoffolder*
```
if `*nameoffolder*` is left empty, the folder name will be HomeworkTeX. After cloning a repository, you need to `cd *nameofrepo*` to enter the directory/repository.

**Checking the status of the repository**

Once inside you can,

```git
git status
```

which shows you the status of that directory/repository. Here directory and repository mean the same thing; it is seen as a directory by the machine but a repository by github.

to be continued...
