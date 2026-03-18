# Contributing

Thank you for your interest in contributing to Model Verify. This document provides guidelines for contributing to the project.

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Test your changes thoroughly
5. Commit with a clear message following the format below
6. Push to your fork: `git push origin feature/my-feature`
7. Open a Pull Request

## Bug Reports

Report bugs using GitHub Issues. Include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS)
- Relevant logs or error messages

## Feature Requests

We welcome feature requests. When suggesting a new feature:

- Explain the use case clearly
- Describe the expected behavior
- Consider if it aligns with the project's goals
- Discuss implementation approach if possible

## Code Style

This project uses Python. Follow these guidelines:

- Use [ruff](https://github.com/astral-sh/ruff) for linting
- Run linting before committing: `ruff check .`
- Format code with: `ruff format .`
- Follow PEP 8 conventions
- Write clear, descriptive variable and function names
- Add docstrings for functions and classes

## Testing

Testing is required for all contributions:

- Run tests with: `pytest`
- Ensure all tests pass before submitting a PR
- Write new tests for new features
- Maintain test coverage for modified code
- Test edge cases and error conditions

## Commit Message Format

Use clear, descriptive commit messages:

```
<type>: <short description>

<optional detailed description>

<optional footer>
```

Types:
- `feat`: new feature
- `fix`: bug fix
- `docs`: documentation changes
- `style`: code style changes (formatting, etc.)
- `refactor`: code refactoring
- `test`: adding or updating tests
- `chore`: maintenance tasks

Examples:
- `feat: add tier signature probe for model differentiation`
- `fix: handle timeout in API client`
- `docs: update README with installation instructions`

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## Getting Help

If you have questions:

- Check existing issues and pull requests
- Read the documentation
- Ask in a new GitHub Issue with the `question` label
