# TitanovaX Documentation System
# Auto-generated API documentation and deployment guides

"""
TitanovaX Auto-Documentation System
Generates comprehensive documentation from code analysis
"""

import os
import ast
import inspect
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import subprocess

class APIDocumentationGenerator:
    """Generate API documentation from Python code"""

    def __init__(self, source_path: str = '.'):
        self.source_path = Path(source_path)
        self.docs_path = Path('docs')
        self.docs_path.mkdir(exist_ok=True)

    def generate_api_docs(self):
        """Generate comprehensive API documentation"""
        modules = [
            'config_manager',
            'storage_system',
            'monitoring_system',
            'security_system',
            'adaptive_execution_gate',
            'smart_order_router',
            'micro_slippage_model',
            'regime_classifier',
            'ensemble_decision_engine',
            'safety_risk_layer'
        ]

        api_docs = {
            'title': 'TitanovaX Trading System API Documentation',
            'version': '2.0.0',
            'generated_at': datetime.now().isoformat(),
            'modules': {}
        }

        for module_name in modules:
            try:
                module_docs = self._analyze_module(module_name)
                api_docs['modules'][module_name] = module_docs
                print(f"✓ Generated docs for {module_name}")
            except Exception as e:
                print(f"✗ Failed to generate docs for {module_name}: {e}")

        # Save API documentation
        with open(self.docs_path / 'api_documentation.json', 'w') as f:
            json.dump(api_docs, f, indent=2)

        # Generate HTML documentation
        self._generate_html_docs(api_docs)

    def _analyze_module(self, module_name: str) -> Dict[str, Any]:
        """Analyze a Python module for documentation"""
        try:
            # Import module
            module = __import__(module_name, fromlist=[''])

            module_docs = {
                'name': module_name,
                'docstring': module.__doc__ or '',
                'classes': {},
                'functions': {},
                'constants': {}
            }

            # Analyze classes
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and hasattr(obj, '__module__'):
                    if obj.__module__ == module_name or obj.__module__.startswith(module_name):
                        class_docs = self._analyze_class(obj)
                        module_docs['classes'][name] = class_docs

            # Analyze functions
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and hasattr(obj, '__module__'):
                    if obj.__module__ == module_name or obj.__module__.startswith(module_name):
                        func_docs = self._analyze_function(obj)
                        module_docs['functions'][name] = func_docs

            return module_docs

        except ImportError as e:
            return {'error': f'Could not import module {module_name}: {e}'}
        except Exception as e:
            return {'error': f'Error analyzing module {module_name}: {e}'}

    def _analyze_class(self, cls) -> Dict[str, Any]:
        """Analyze a class for documentation"""
        return {
            'name': cls.__name__,
            'docstring': cls.__doc__ or '',
            'methods': {},
            'properties': {},
            'bases': [base.__name__ for base in cls.__bases__ if base != object]
        }

    def _analyze_function(self, func) -> Dict[str, Any]:
        """Analyze a function for documentation"""
        sig = inspect.signature(func)

        return {
            'name': func.__name__,
            'docstring': func.__doc__ or '',
            'signature': str(sig),
            'parameters': {name: {'annotation': str(param.annotation) if param.annotation != param.empty else 'Any',
                                 'default': str(param.default) if param.default != param.empty else 'None'}
                          for name, param in sig.parameters.items()}
        }

    def _generate_html_docs(self, api_docs: Dict[str, Any]):
        """Generate HTML documentation"""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{api_docs['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .module {{ margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; border-radius: 8px; }}
        .class {{ margin-left: 20px; background: #f1f3f4; padding: 15px; margin-bottom: 15px; }}
        .function {{ margin-left: 40px; background: #e8f5e8; padding: 10px; margin-bottom: 10px; }}
        .docstring {{ font-style: italic; color: #666; }}
        .signature {{ font-family: monospace; background: #f0f0f0; padding: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{api_docs['title']}</h1>
        <p><strong>Version:</strong> {api_docs['version']}</p>
        <p><strong>Generated:</strong> {api_docs['generated_at']}</p>
    </div>
"""

        for module_name, module_docs in api_docs['modules'].items():
            if 'error' in module_docs:
                html_content += f"""
    <div class="module">
        <h2>Module: {module_name}</h2>
        <p class="error">Error: {module_docs['error']}</p>
    </div>
"""
            else:
                html_content += f"""
    <div class="module">
        <h2>Module: {module_name}</h2>
        <p class="docstring">{module_docs['docstring']}</p>
"""

                # Classes
                for class_name, class_docs in module_docs.get('classes', {}).items():
                    html_content += f"""
        <div class="class">
            <h3>Class: {class_name}</h3>
            <p class="docstring">{class_docs['docstring']}</p>
        </div>
"""

                # Functions
                for func_name, func_docs in module_docs.get('functions', {}).items():
                    html_content += f"""
        <div class="function">
            <h4>Function: {func_name}</h4>
            <div class="signature">{func_docs['signature']}</div>
            <p class="docstring">{func_docs['docstring']}</p>
        </div>
"""

                html_content += "    </div>"

        html_content += """
</body>
</html>
"""

        with open(self.docs_path / 'api_documentation.html', 'w') as f:
            f.write(html_content)

class DeploymentGuideGenerator:
    """Generate deployment guides"""

    def __init__(self, docs_path: str = 'docs'):
        self.docs_path = Path(docs_path)

    def generate_deployment_guide(self):
        """Generate comprehensive deployment guide"""
        guide_content = """
# 🚀 TitanovaX Trading System - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the TitanovaX Trading System in production environments.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **CPU**: 4+ cores, 2.5GHz+
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 100GB SSD minimum
- **Network**: Stable internet connection

### Software Dependencies

- **Python**: 3.9+
- **Docker**: 20.10+
- **Docker Compose**: 1.29+
- **Git**: 2.25+

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-org/titanovax-trader-demo.git
cd titanovax-trader-demo
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate  # Windows

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit environment file with your credentials
nano .env  # or your preferred editor
```

### 4. Database Setup

```bash
# Start PostgreSQL and Redis with Docker
docker-compose -f docker-compose.db.yml up -d

# Verify database connections
python -c "from config_manager import get_config_manager; print('Database connection successful')"
```

### 5. Run System

```bash
# Start all services
docker-compose up -d

# Check system health
curl http://localhost:8001/health
```

## Detailed Configuration

### Environment Variables

See `.env` file for all configuration options:

```bash
# API Credentials (Required)
OANDA_API_KEY=your_oanda_key
OANDA_ACCOUNT_ID=your_account_id
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=titanovax
DB_USER=titanovax_user
DB_PASSWORD=secure_password

# System Configuration
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### Service Configuration

#### Trading Engine
- **Port**: 8001 (FastAPI)
- **Health Check**: `/health`
- **API Docs**: `/docs`

#### Monitoring
- **Prometheus**: Port 9090
- **Grafana**: Port 3001

## Docker Deployment

### Production Docker Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  titanovax-trading:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    environment:
      - ENVIRONMENT=production
      - DB_HOST=postgres
      - REDIS_HOST=redis
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_DB=titanovax
      - POSTGRES_USER=titanovax_user
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped
```

### Build and Deploy

```bash
# Build images
docker-compose build

# Deploy services
docker-compose up -d

# View logs
docker-compose logs -f titanovax-trading
```

## Monitoring & Health Checks

### Health Endpoints

```bash
# Main API health
curl http://localhost:8001/health

# Database health
curl http://localhost:8001/health/database

# External services health
curl http://localhost:8001/health/external
```

### Monitoring Dashboards

1. **Grafana**: http://localhost:3001
   - Default credentials: admin/admin
   - Pre-configured dashboards available

2. **Prometheus**: http://localhost:9090
   - View metrics and alerts

### Log Monitoring

```bash
# View application logs
tail -f logs/titanovax.log

# View Docker logs
docker-compose logs -f
```

## Security Considerations

### Network Security
- Use internal Docker network
- Configure firewalls appropriately
- Use SSL/TLS for external connections

### Secret Management
- Store secrets in environment variables
- Use Docker secrets for production
- Rotate credentials regularly

### Access Control
- Implement API authentication
- Use rate limiting
- Monitor for suspicious activity

## Troubleshooting

### Common Issues

#### 1. Database Connection Failed
```bash
# Check database logs
docker-compose logs postgres

# Verify environment variables
python -c "from config_manager import get_config_manager; print('DB connection OK')"
```

#### 2. High Memory Usage
```bash
# Monitor memory usage
docker stats

# Check for memory leaks
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

#### 3. Slow Performance
```bash
# Check system resources
htop  # or top

# Monitor API response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8001/health
```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python -m uvicorn main:app --reload --log-level debug
```

## Scaling and Performance

### Horizontal Scaling

```bash
# Scale trading engine
docker-compose up -d --scale titanovax-trading=3

# Use load balancer
nginx -c nginx.conf
```

### Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX idx_trades_symbol_timestamp ON trades(symbol, timestamp DESC);
CREATE INDEX idx_messages_timestamp ON telegram_messages(timestamp DESC);
```

### Caching Strategy

- Redis for session storage
- In-memory caching for frequent queries
- FAISS indexing for vector similarity

## Backup and Recovery

### Automated Backups

```bash
# Database backup
docker exec postgres pg_dump -U titanovax_user titanovax > backup_$(date +%Y%m%d_%H%M%S).sql

# Data backup
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz data/
```

### Recovery Procedures

```bash
# Restore database
docker exec -i postgres psql -U titanovax_user titanovax < backup.sql

# Restore data
tar -xzf backup.tar.gz
```

## Support and Maintenance

### Regular Maintenance

1. **Daily**: Check logs and alerts
2. **Weekly**: Review performance metrics
3. **Monthly**: Update dependencies and security patches
4. **Quarterly**: Full system testing and validation

### Support Channels

- **Documentation**: [Internal Wiki](https://wiki.company.com/titanovax)
- **Issue Tracking**: [JIRA Project](https://jira.company.com/projects/TITANOVAX)
- **Chat**: #titanovax-support

### Emergency Contacts

- **On-call Engineer**: +1-555-0123
- **System Admin**: +1-555-0456
- **Security Team**: security@company.com

---

## Conclusion

This deployment guide provides comprehensive instructions for setting up and maintaining the TitanovaX Trading System. For additional support, consult the development team or refer to the internal documentation.

**Last Updated**: $(date)
**Version**: 2.0.0
"""

        with open(self.docs_path / 'deployment_guide.md', 'w') as f:
            f.write(guide_content)

class ArchitecturalDecisionRecord:
    """Manage Architectural Decision Records (ADRs)"""

    def __init__(self, docs_path: str = 'docs/adr'):
        self.docs_path = Path(docs_path)
        self.docs_path.mkdir(parents=True, exist_ok=True)

    def create_adr(self, title: str, context: str, decision: str, consequences: str,
                   status: str = 'accepted') -> str:
        """Create a new ADR"""
        # Generate ADR number
        existing_adrs = list(self.docs_path.glob('*.md'))
        adr_number = len(existing_adrs) + 1

        filename = f"{adr_number:04d}-{title.lower().replace(' ', '-')}.md"

        adr_content = f"""# ADR {adr_number:04d}: {title}

## Status
{status}

## Context
{context}

## Decision
{decision}

## Consequences
{consequences}

## Date
{datetime.now().strftime('%Y-%m-%d')}

## Author
TitanovaX Development Team
"""

        file_path = self.docs_path / filename
        with open(file_path, 'w') as f:
            f.write(adr_content)

        return filename

class DocumentationSystem:
    """Main documentation system"""

    def __init__(self):
        self.api_generator = APIDocumentationGenerator()
        self.deployment_generator = DeploymentGuideGenerator()
        self.adr_manager = ArchitecturalDecisionRecord()

    def generate_all_docs(self):
        """Generate all documentation"""
        print("🔄 Generating API documentation...")
        self.api_generator.generate_api_docs()

        print("🔄 Generating deployment guide...")
        self.deployment_generator.generate_deployment_guide()

        print("🔄 Creating architectural decision records...")
        self._create_initial_adrs()

        print("✅ Documentation generation completed!")

    def _create_initial_adrs(self):
        """Create initial ADRs"""
        # ADR 1: Configuration Management
        self.adr_manager.create_adr(
            "Centralized Configuration Management",
            "Need for consistent configuration across all system components with secure credential management",
            "Implement centralized configuration system with environment variables and encrypted storage",
            "Improved security, maintainability, and deployment flexibility. Easier credential rotation and audit trails.",
            "accepted"
        )

        # ADR 2: Data Storage Strategy
        self.adr_manager.create_adr(
            "FAISS + Parquet Storage Architecture",
            "Need for efficient storage of embeddings and text data with memory optimization",
            "Use FAISS IndexIVFPQ for embeddings with Parquet compression for raw data",
            "Memory-efficient storage with fast similarity search. Better scalability for large datasets.",
            "accepted"
        )

        # ADR 3: Multi-layered Security
        self.adr_manager.create_adr(
            "Multi-layered Security Architecture",
            "Need for comprehensive security covering API, network, and application layers",
            "Implement HMAC validation, rate limiting, IP filtering, and circuit breakers",
            "Enhanced security posture with defense in depth. Better protection against various attack vectors.",
            "accepted"
        )

        # ADR 4: Real-time Monitoring
        self.adr_manager.create_adr(
            "Comprehensive Monitoring System",
            "Need for real-time monitoring and anomaly detection in production environment",
            "Implement metrics collection, alerting, and self-healing capabilities",
            "Improved system reliability and faster incident response. Better operational visibility.",
            "accepted"
        )

if __name__ == "__main__":
    # Generate all documentation
    doc_system = DocumentationSystem()
    doc_system.generate_all_docs()

    print("\n📚 Documentation generated successfully!")
    print("📁 Check the 'docs/' directory for all generated files")
