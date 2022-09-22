################ This script is customized for use on PNNL's internal CI platform
echo "Pushing temp develop branch to Github"

############
#### Get all the required repo data from Stash

# Get all the branches, tags
git fetch --all

# Checkout the appropriate branch
git checkout develop

# log the branch, remotes, and last commit for forsensics
git branch -vv
git remote -vv
git --no-pager show --name-only
############

############
#### Setup access to HyperNetX repo on pnnl-public Github

# Add the HyperNetX Github remote to list of local remotes
# to access the HyperNetX Github repo
git remote add gh-origin "https://${bamboo.GITHUB_USERNAME}:${bamboo.GHUB_PAT_PASSWORD}@github.com/pnnl/HyperNetX.git"

# log the remotes for forsensics
git remote -vv

# While on the 'develop' branch from Stash,
# create a unique branch name using the shortened commit hash
# of the last commit on 'develop'
# example: develop-c424244
temp_develop=develop-$(git rev-parse --short HEAD)
# IMPORTANT: Write the branch name to a file so that we can save it on the Docker image
echo "${temp_develop}" > tempbranch.txt
cat tempbranch.txt

# Push the temporary branch to HyperNetXGithub repo
# so that we can create an PR off of this branch on Github using the Docker image
git push gh-origin develop:"${temp_develop}"

# Write the Github PAT to a text file so that we can save it on the Docker image
echo "${bamboo.GHUB_PAT}" > ghtoken.txt
############

############
#### Docker prep work
# Remove any docker containers named "hnx"
# so that we can start and create a new one
docker stop hnx || true
docker rm hnx || true
############
