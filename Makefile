.PHONY: test cover clean lint create_env

#################################################################################
# GLOBALS                                                                       #
#################################################################################

REPO_NAME = pyomeca
EXCLUDES_LINT = --exclude=bin/,src/rebuydsutils/,docs/conf.py
EXCLUDES_PYTEST = --ignore src/rebuydsutils
SHELL=/bin/bash

ifeq (,$(shell which conda))
	$(error conda must be installed)
endif

# Define utility variable to help calling Python from the virtual environment
ifeq ($(CONDA_DEFAULT_ENV),$(REPO_NAME))
    ACTIVATE_ENV := true
else
    ACTIVATE_ENV := source activate $(REPO_NAME)
endif

# Execute python related functionalities from within the project's environment
define execute_in_env
	$(ACTIVATE_ENV) && $1
endef

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Run pytest on the project
test:
	$(call execute_in_env, python -m pytest --color=yes tests $(EXCLUDES_PYTEST))

## Run coverage test on the project
cover:
	$(call execute_in_env, python -m pytest --color=yes --cov=pyomeca tests $(EXCLUDES_PYTEST))

## Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;

## Lint using flake8; Excluding $EXCLUDES_LINT
lint:
	$(call execute_in_env, flake8  $(EXCLUDES_LINT) .)

## Set up python interpreter environment
create_env:
	conda env create -n $(REPO_NAME) -f environment.yml
	rm -rf *.egg-info

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
