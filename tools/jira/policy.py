from config import (
    Settings,
)

from services.jira.project_mutations import (
    build_jira_project_mutation_service,
)


def evaluate_jira_project_create_policy(
    project_key: str,
    project_name: str,
    template: str,
):
    """
    Trusted pre-approval policy.

    This policy performs read-only provider validation and binds
    provider-owned execution state into the approval snapshot.

    It does not execute the mutation.
    """

    service = (
        build_jira_project_mutation_service(
            Settings()
        )
    )

    return (
        service.prepare_create_project(
            project_key=(
                project_key
            ),

            project_name=(
                project_name
            ),

            template=(
                template
            ),
        )
    )

def evaluate_jira_project_update_policy(
    project_id_or_key: str,
    new_name: str,
):
    """
    Trusted read-only pre-approval policy for project rename.

    The policy freezes the exact provider project identity and
    old project name into the durable approval snapshot.
    """

    service = (
        build_jira_project_mutation_service(
            Settings()
        )
    )

    return (
        service.prepare_update_project(
            project_id_or_key=(
                project_id_or_key
            ),

            new_name=(
                new_name
            ),
        )
    )

def evaluate_jira_project_archive_policy(
    project_id_or_key: str,
):
    """
    Trusted read-only pre-approval policy for Jira project archive.
    """

    service = (
        build_jira_project_mutation_service(
            Settings()
        )
    )

    return (
        service.prepare_archive_project(
            project_id_or_key=(
                project_id_or_key
            )
        )
    )

