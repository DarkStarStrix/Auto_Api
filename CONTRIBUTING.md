# Contributing to AutoML API

## Setting up Development Environment

1. Clone the repository:
```bash
git clone https://github.com/yourusername/automl-api.git
cd automl-api
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. Set up pre-commit hooks:
```bash
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test categories
pytest -m "not slow"
```

## Code Style

- Use Black for code formatting
- Sort imports with isort
- Follow PEP 8 guidelines
- Write docstrings in Google style

## Pull Request Process

1. Create a feature branch from `develop`
2. Make your changes
3. Run tests locally
4. Push your changes
5. Create a PR to `develop`

## CI/CD Pipeline

The CI/CD pipeline includes:
1. Code quality checks
2. Security scanning
3. Testing
4. Building and pushing Docker image
5. Deployment

### Required Secrets

Set these in GitHub repository settings:
- `JWT_SECRET_KEY`
- `VALID_API_KEY`
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `SSH_HOST`
- `SSH_USERNAME`
- `SSH_PRIVATE_KEY`
