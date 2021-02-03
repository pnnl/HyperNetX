## Notes On Everything I find useful but am apt to forget:

### Step 1 - you need to be in bash and link to the system's anaconda3 library
On puma:
I put references to anaconda3 in my bash account. That way you can use existing 
anaconda packages already available on puma:

Start by going on puma using port forwarding:
```
ssh -L 8192:localhost:8192 -Y puma.pnl.gov
```
And login as usual. You will need the port forwarding for using Jupyter, if you don't intend to use Jupyter you can just use
```
ssh puma
```


On the command line in puma:
```
cd   ## put yourself into your home directory if you aren't already there
bash ## assuming bash isn't your default shell
nano .bashrc  ## if you don't have a .bashrc file this will create it
```

In your .bashrc file make sure you have these lines:
```
# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# User specific aliases and functions
 . /share/apps/python/anaconda3/etc/profile.d/conda.sh
# conda activate

export PATH=$PATH:/share/apps/python/anaconda3/lib
```

On your command line run the .bashrc file by restarting the shell:
```
bash
```

From now on when you get on puma, you will just start with:
```
bash
```

### Step 2:
Create a conda environment to use with HNX:
```
conda create --name hnxenv python=3.8 
conda activate hnxenv
```
If you haven't loaded hnx yet:

