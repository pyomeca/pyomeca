All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.
We recommend going through the [`issues`](https://github.com/pyomeca/pyomeca/issues) to find issues that interest you.
Then, you can get your development environment setup with the following instructions.

## Forking Pyomeca

You will need your own fork to work on the code.
Go to the [pyomeca project page](https://github.com/pyomeca/pyomeca/) and hit the `Fork` button.
You will want to clone your fork to your machine:

```bash
git clone https://github.com/your-user-name/pyomeca.git
cd pyomeca
git remote add upstream https://github.com/pyomeca/pyomeca.git
```

## Creating a Python environment

Before starting any development, you’ll need to create an isolated pyomeca development environment:

- Install [miniconda](https://conda.io/miniconda.html)
- `cd` to the pyomeca source directory
- Install pyomeca dependencies with:

```bash
conda env create
conda activate pyomeca
pip install -r requirements-dev.txt
```

## Creating a branch

You want your master branch to reflect only production-ready code, so create a feature branch for making your changes.
For example:

```bash
git branch -b new-feature
```

Keep any changes in this branch specific to one bug or feature so it is clear what the branch brings to pyomeca.

## Testing your code

Adding tests is required if you add or modify existing codes in pyomeca.
Therefore, it is worth getting in the habit of writing tests ahead of time so this is never an issue.
The pyomeca test suite runs automatically on GitHub Actions, once your pull request is submitted.
However, we strongly encourage running the tests prior to submitting the pull request.
To do so, simply run `make test`.

## Linting your code

Pyomeca uses several tools to ensure a consistent code format throughout the project.
The easiest way to use them is to run `make lint` from the source directory.

## Making the pull-request

When you want your changes to appear publicly on your GitHub page, push your forked feature branch’s commits:

```bash
git push origin new-feature
```

If everything looks good, you are ready to make a pull request.
This pull request and its associated changes will eventually be committed to the master branch and available in the next release.

1. Navigate to your repository on GitHub
2. Click on the `Pull Request` button
3. You can then click on `Files Changed` to make sure everything looks OK
4. Write a description of your changes in the Discussion tab
5. Click `Send Pull Request`

This request then goes to the repository maintainers, and they will review the code.
If you need to make more changes, you can make them in your branch, add them to a new commit and push them to GitHub.
The pull request will be automatically updated.

!!! note "PR Checklist"

    Let's summarize the steps needed to get your PR ready for submission.
    0. **Use an isolated Python environment**.
    1. **Properly test your code**. Write new tests if needed and make sure that your modification didn't break anything by running `make test`.
    2. **Properly format your code**. You can verify that it passes the formatting guidelines by running `make lint`.
    3. **Push your code and create a PR**.
    4. **Properly describe your modifications** with a helpful title and description. If this addresses an issue, please reference it.
