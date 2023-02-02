################ This script is customized for use on PNNL's internal CI platform
# At this point in our CD pipeline, two manual actions have already been performed on the Stash repo (source of truth) by the release engineer:
# 1) An official release branch (i.e. release-X.X) has already been merged into master by the release engineer
# 2) A tag has been created (i.e. vX.X)
echo "Publishing to PyPi, pushing tag to Github"


############
#### Get all the required repo data from Stash

# Get all the branches, tags from Stash
git fetch --all

# Checkout the latest tag
## get all the tags from Stash, filter tags that only begin with v and a number, sort the tags in reverse order (from highest to lowest), then get the first (i.e. most current tag)
## uses -V which is version sort to keep it monotonically increasing.
current_tag=$(git tag | grep '^v[0-9]' | sort --reverse -V  | sed -n 1p)
echo "${current_tag}"
git checkout "${current_tag}"

# log the branch, remotes, and last commit for forensics
git --no-pager branch -vv
git --no-pager remote -vv
# show the last commit on checked out branch
git --no-pager show --name-only
############




############
echo "Publishing to PyPI"

# Assumes the following environment variables are set: TWINE_USERNAME, TWINE_PASSWORD, TWINE_REPOSITORY_URL
make build-dist
make publish-to-pypi
############



############
echo "Pushing release tag to Github"

# Setup access to HyperNetX repo on pnnl-public Github
# Add the HyperNetX Github remote to list of local remotes
# to access the HyperNetX Github repo
git remote add origin-github "https://${bamboo.GHUB_USERNAME}:${bamboo.GHUB_PAT}@github.com/pnnl/HyperNetX.git"

# log the remotes for forensics
git remote -vv

# Push the latest tag branch to HyperNetX repo on Github
git push origin-github "${current_tag}"
############
