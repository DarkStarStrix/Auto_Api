# Contributing to AutoML Pipeline

## Ways to Contribute
1. Adding Configuration Templates
2. Improving Core Functionality
3. Documentation
4. Bug Reports
5. Feature Requests

## Configuration Templates
### Adding New Templates
1. Create new template in `config.py`
2. Include documentation
3. Add example usage
4. Submit PR

Example:
```python
def get_vision_config():
    """
    Configuration template for computer vision tasks.
    
    Returns:
        dict: Configuration dictionary
    """
    return {
        "model": {
            "type": "vision",
            "architecture": "resnet18"
        }
        # ... other parameters
    }
```

## Core Development
### Setting Up Development Environment
```bash
git clone https://github.com/your-username/automl-pipeline
cd automl-pipeline
pip install -e ".[dev]"
```

### Running Tests
```bash
pytest tests/
```

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings
- Keep functions focused

## Pull Request Process
1. Create feature branch
2. Add tests
3. Update documentation
4. Submit PR
5. Address reviews

## API Development
### Endpoint Structure
```
/api/v1/
├── models/
├── configs/
└── training/
```

### Adding New Endpoints
1. Create endpoint in `api/`
2. Add tests
3. Update API documentation
4. Submit PR

## Documentation
### Adding Tutorials
1. Create markdown file in `docs/tutorials/`
2. Include:
   - Overview
   - Code examples
   - Results
   - Visualizations
3. Submit PR
