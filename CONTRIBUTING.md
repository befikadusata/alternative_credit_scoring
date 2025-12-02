# Contributing to the Alternative Credit Scoring Platform

This document outlines the branching strategy and workflow for contributing to this project. Following these guidelines ensures a consistent, organized, and high-quality codebase.

## Branching Strategy: GitHub Flow

For this project, we utilize a simplified branching strategy known as **GitHub Flow**. This approach is ideal for projects that practice continuous integration and deployment, ensuring the `main` branch is always stable and deployable.

### Core Principles

1.  **`main` is always deployable:** The `main` branch is the single source of truth for the project. Code on `main` is always expected to be stable, fully tested, and ready for deployment at any time.
2.  **Feature branches for all work:** All new development, bug fixes, or even minor changes should be done on dedicated feature branches, not directly on `main`.
3.  **Pull Requests (PRs) for collaboration:** Changes made on feature branches are integrated into `main` via Pull Requests, which serve as a forum for code review and automated checks.

### Workflow

Follow these steps when contributing to the project:

#### 1. Create a New Branch

Before starting any work (feature, bugfix, documentation update), create a new, descriptively named branch from the `main` branch.

```bash
# First, ensure your main branch is up-to-date
git checkout main
git pull origin main

# Then, create your new branch
git checkout -b <your-branch-name>
```

**Branch Naming Conventions:**
*   Use lowercase, hyphen-separated words.
*   Prefix with the type of work:
    *   `feature/descriptive-name`: For new features or significant enhancements.
    *   `bugfix/issue-description`: For bug fixes.
    *   `docs/document-update`: For documentation changes.
    *   `refactor/module-name`: For code refactoring.
    *   `chore/task-description`: For maintenance tasks (e.g., dependency updates).

    *Example:* `feature/add-batch-prediction` or `bugfix/api-auth-error`

#### 2. Make Your Changes

Implement your code changes, write or update tests, and update relevant documentation. Ensure your changes are focused on the purpose of your branch.

#### 3. Commit Your Work

Commit your changes frequently and with clear, descriptive commit messages. A good commit message explains *what* was changed and *why*.

```bash
git add .
git commit -m "feat: Implement the batch prediction endpoint"
```

#### 4. Push Your Branch

Regularly push your changes to your remote branch on GitHub. This acts as a backup and allows others (if applicable) to see your progress.

```bash
git push origin <your-branch-name>
```

#### 5. Open a Pull Request (PR)

Once your work is complete and tested locally, navigate to GitHub and open a Pull Request from your feature branch to the `main` branch.

*   **Provide a clear title and description** for your PR, explaining the changes and their purpose.
*   **Reference any related issues** (e.g., "Closes #123").
*   **Ensure all automated checks pass:** The CI pipeline will automatically run linting and tests. Your PR cannot be merged until these pass.
*   (Self-review) Even as a solo developer, use the PR as an opportunity for a final review of your own code.

#### 6. Merge the Pull Request

After all checks pass and you are satisfied with the changes, merge your PR into `main`.

*   Use a "Squash and Merge" option if you want to consolidate multiple small commits into a single, clean commit on `main`.
*   Merging to `main` signifies that the code is now stable and potentially ready for deployment.

#### 7. Delete Your Branch

After your PR is merged, delete your feature branch to keep the repository clean and tidy. GitHub usually offers this option automatically after a merge.

---

By adhering to this GitHub Flow, we maintain a clean, organized, and reliable codebase that is always ready for further development or deployment.
