"""Module with tests tasks"""

import invoke


@invoke.task
def unit_tests(context):
    """Run unit tests

    :param context: invoke.Context instance
    """

    context.run("pytest ./tests", pty=True, echo=True)


@invoke.task
def static_code_analysis(context):
    """Run static code analysis

    :param context: invoke.Context instance
    """

    directories = "net tests"

    context.run("pycodestyle {}".format(directories), echo=True)
    context.run("xenon . --max-absolute B", echo=True)
    context.run("mypy .", echo=True)
    context.run("pylint {}".format(directories), echo=True)


@invoke.task
def commit_stage(context):
    """
    Run commit stage tasks

    :param context: invoke.Context instance
    """

    unit_tests(context)
    static_code_analysis(context)


@invoke.task
def inserts_count_check(context):
    """
    Check current tree doesn't have too many changes w.r.t. origin/master

    :param context: invoke.Context instance
    """

    import git

    def should_modification_be_ignored(path):
        """
        Simple helper for filtering out git modifications that shouldn't be counted towards insertions check.
        Filters out tools configuration files and similar.

        :param path: str, path of file that was modified
        :return: bool
        """

        # Likely a file was deleted
        if path is None:
            return True

        patterns = [
            ".devcontainer",
            ".pylintrc",
            ".gitignore"
        ]

        for pattern in patterns:

            if pattern in path:

                return True

        return False

    repository = git.Repo(".")
    repository.remote().fetch()

    master = repository.commit("remotes/origin/master")

    additions_count = 0

    # compare origin/master to working tree
    for diff_object in repository.commit(master.hexsha).diff(other=None, create_patch=True):

        # Only look at inserts for files that shouldn't be ignored
        if should_modification_be_ignored(path=diff_object.b_path) is False:

            changed_lines = diff_object.diff.decode('utf-8').split('\n')
            additions = ([line for line in changed_lines if len(line) > 0 and line[0] == "+"])

            additions_count += len(additions)

    threshold = 300

    print(f"Inserts between origin/master and HEAD: {additions_count}/{threshold}")

    if additions_count > threshold:

        raise ValueError("Exceeded max inserts count")


@invoke.task
def mlflow_experiment(_context, config_path):
    """
    A very simple mlflow logging example

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import mlflow

    import net.utilities

    config = net.utilities.read_yaml(config_path)

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment("simple_experiment")

    with mlflow.start_run(run_name="simple_run"):

        print("started run")

        for index in range(10):
            print(f"logging index {index}")

            mlflow.log_metric("index", index)
            mlflow.log_metric("index square", index ** 2)
            mlflow.log_metric("index low", 1 / (1 + index))
