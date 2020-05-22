.PHONY : test lint doc nb_to_md clean all

lint:
	black .

test:
	pytest --cov-report term-missing --color=yes --cov=pyomeca tests -rxXs

nb_to_md:
	jupyter nbconvert --to markdown notebooks/getting-started.ipynb --output-dir='./docs' --template=docs/nbconvert.tpl

doc:
    # copy readme, correct path and delete link to documentation
	sed 's,docs/,,g' README.md > docs/index.md; \
    sed -i -z "s,\n## Pyomeca documentation\n\nSee Pyomeca's \[documentation site\](https://pyomeca.github.io).\n,,g" docs/index.md; \
    sed -i -z "s,\nSee \[the documentation\](https://pyomeca.github.io) for more details and examples.\n,,g" docs/index.md; \
	cd ../pyomeca.github.io; \
	mkdocs gh-deploy --config-file ../pyomeca/mkdocs.yml --remote-branch master
	rm -rf site

clean:
	rm -rf .pytest_cache .coverage site notebooks/.ipynb_checkpoints

all: lint nb_to_md test doc clean
