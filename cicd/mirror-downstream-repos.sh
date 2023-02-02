############
#### Get all the required repo data from Stash

# Get all the branches, tags
git fetch --all

# Checkout the appropriate branch
git checkout master

# log the branch, remotes, and last commit for forensics
git --no-pager branch -vv
git --no-pager remote -vv
# show the last commit on checked out branch
git --no-pager show --name-only
############



############
#### Setup access to HyperNetX repo on pnnl-public Github

# Add the HyperNetX Github remote to list of local remotes
git remote add origin-github "https://${bamboo.GHUB_USERNAME}:${bamboo.GHUB_PAT}@github.com/pnnl/HyperNetX.git"

# Add HyperNetX Gitlab remote to list of local remotes
git remote add origin-gitlab "https://${bamboo.GLAB_USERNAME}:${bamboo.GLAB_PSSWD}@gitlab.pnnl.gov/hyp/hypernetx.git"

# log the remotes for forsensics
git remote -vv

# Update Github repo
git push origin-github master

# Update Gitlab repo on branch develop
git push origin-gitlab master
