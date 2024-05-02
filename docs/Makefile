# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC ?= sphinx-apidoc
SPHINXAUTOBUILD ?= sphinx-autobuild
PAPER         =
BUILDDIR      = build
SOURCEDIR	  = source

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Make sure you have Sphinx installed, then set the SPHINXBUILD environment variable to point to the full path of the '$(SPHINXBUILD)' executable. Alternatively you can add the directory with the executable to your PATH. If you don't have Sphinx installed, grab it from http://sphinx-doc.org/)
endif

# Internal variables.
#PAPEROPT_a4     = -D latex_paper_size=a4
#PAPEROPT_letter = -D latex_paper_size=letter
#ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source
# the i18n builder cannot share the environment and doctrees with the others
#I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html: clean
	$(SPHINXAPIDOC) --module-first -f -o "$(SOURCEDIR)"/classes ../hypernetx/classes
	$(SPHINXAPIDOC) -f -o "$(SOURCEDIR)"/algorithms ../hypernetx/algorithms
	$(SPHINXAPIDOC) -f -o "$(SOURCEDIR)"/drawing ../hypernetx/drawing
	$(SPHINXAPIDOC) -f -o "$(SOURCEDIR)"/reports ../hypernetx/reports
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

livehtml:
	$(SPHINXAUTOBUILD) $(SOURCEDIR) $(BUILDDIR)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile livehtml html
