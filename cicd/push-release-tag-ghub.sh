################ This script is customized for use on PNNL's internal CI platform
echo "Pushing release tag to Github"

############
#### Get all the required repo data from Stash

# Get all the branches, tags
git fetch --all

# Checkout the latest tag
## get all the tags from Stash, filter tags that only begin with v and a number, sort the tags in reverse order (from highest to lowest), then get the first (i.e. most current tag)
## uses -V which is version sort to keep it monotonically increasing.
current_tag=$(git tag | grep '^v[0-9]' | sort --reverse -V  | sed -n 1p)
echo "${current_tag}"
git checkout "${current_tag}"
# IMPORTANT: Write the branch name to a file so that we can save it on the Docker image
echo "${current_tag}" > tempbranch.txt
cat tempbranch.txt

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

# Push the latest tag branch to HyperNetX repo on Github
# so that we can create an PR off of this branch on Github using the Docker image
git push gh-origin "${current_tag}"

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
