# HomeworkTeX
Homework that are submitted in tex inhabit this repository. Do not disturb!!!


## Requirements
If you like to collaborate on assignments with me but have no knowledge of Git or github, below are some instructions that might be helpful for you.

- a Git is a version control system that allows multiple parties to collaborate on a single project/document (in this case the tex file) without ruining each other's version.

- github is a Dropbox like freemium which is able to do version control at the same time. You can signup for a github account [here](https://github.com/)

- to start collaborating the Git way, you will need a distribution of Git on your machine.
  - for Windows users I suggest getting the Git for Windows Portable [32bit](https://github.com/git-for-windows/git/releases/download/v2.10.1.windows.1/PortableGit-2.10.1-32-bit.7z.exe)/[64bit](https://github.com/git-for-windows/git/releases/download/v2.10.1.windows.1/PortableGit-2.10.1-64-bit.7z.exe)
  - for Mac users, one can do the installation by following the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)


## How to Git?

Assuming that you have a github account and Git distribution on your machine we are ready to start! Start up your terminal or PortableGit for Mac and Windows system respectively. We will start by cloning this repository. The commands for terminal and PortableGit work the same

**Basics**
```git
pwd: tells you the present working directory.
cd: goes back to home directory
cd ..: jumps back one previous directory.
ls: list the files in the directory you are in.
```
Tip: `pwd` is used to check which directory you are in, `cd` to change to the directory that you want to access, `ls` is for you to check if the file you are trying to access is in that directory. Use them to navigate around the directories and find your files :)

**Cloning this repository**

```git

git clone https://github.com/zunction/HomeworkTeX *nameoffolder*
```
if `*nameoffolder*` is left empty, the folder name will be HomeworkTeX. After cloning a repository, you need to `cd *nameofrepo*` to enter the directory/repository. Once inside you can

```git
git status
```

which shows you the status of that directory/repository. Here directory and repository mean the same thing; it is seen as a directory by the machine but a repository by github.

to be continued...
