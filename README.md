# AGIcommander

**An AI-powered autonomous development assistant that can understand, modify, and improve code at the project level.**

AGIcommander evolved from the original Commander tool and represents a significant step toward Artificial General Intelligence (AGI) for software development. It uses the Model Context Protocol (MCP) to coordinate between multiple AI models and specialized servers, enabling autonomous code improvement and self-directed learning.

## 🚀 Features

### **Project-Level AI Processing**
- **Multi-file analysis** across entire codebases
- **Intelligent code modification** with safety controls
- **Autonomous improvement suggestions** based on best practices
- **Cross-language support** (Python, JavaScript, HTML, CSS, JSON, Markdown, and more)

### **Self-Improving Architecture**
- **MCP-based modular design** for extensibility
- **Multiple LLM provider support** (Gemini, GPT, Claude, Grok)
- **Specialized AI agents** for different development tasks
- **Safety-first approach** with human oversight controls

### **Advanced Capabilities**
- **Autonomous learning cycles** for continuous improvement
- **Vector database memory** for long-term knowledge retention
- **Real-time web research** integration
- **Git workflow automation** with pull request generation

## 🏗️ Architecture

```
AGIcommander/
├── core/                    # Central orchestration and safety controls
├── llm_adapters/           # Unified interface for AI providers
├── servers/                # MCP servers for specialized capabilities
│   ├── code/              # Code analysis and modification
│   ├── learning/          # Research and knowledge acquisition
│   ├── memory/            # Data persistence and retrieval
│   └── self_reflection/   # Self-analysis and improvement
├── agents/                # Specialized AI agents
└── config/               # Configuration and safety rules
```

## 🔧 Quick Start

### **Prerequisites**
- Python 3.8+
- Google API key for Gemini
- Git repository (for workflow features)

### **Installation**

1. **Clone and setup:**
   ```bash
   git clone <your-repo-url>
   cd agicommander
   python setup.py
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

3. **Activate virtual environment:**
   ```bash
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

4. **Test the installation:**
   ```bash
   python test_basic.py
   ```

### **Basic Usage**

Create a `commander.txt` file with your instructions:
```
Add comprehensive error handling to all Python functions.
Include proper logging and user-friendly error messages.
```

Then run:
```bash
python -c "
import asyncio
from core.orchestrator import AGIOrchestrator

async def main():
    orchestrator = AGIOrchestrator()
    await orchestrator.initialize()
    result = await orchestrator.execute_task('Process files according to commander.txt')
    print(result)
    await orchestrator.shutdown()

asyncio.run(main())
"
```

## 📖 Examples

### **Code Analysis**
```python
await orchestrator.execute_task(
    "Analyze the codebase for security vulnerabilities and suggest fixes"
)
```

### **Automated Refactoring**
```python
await orchestrator.execute_task(
    "Refactor large functions into smaller, more maintainable components"
)
```

### **Documentation Generation**
```python
await orchestrator.execute_task(
    "Generate comprehensive API documentation for all public functions"
)
```

## 🛡️ Safety Features

AGIcommander includes robust safety mechanisms:

- **Human approval workflows** for sensitive operations
- **Sandboxed execution** environments
- **Rate limiting** and operation monitoring
- **Audit trails** for all modifications
- **Rollback capabilities** for failed changes
- **Path validation** to protect system directories

## 🔮 Future Vision

AGIcommander is designed as a stepping stone toward AGI development:

### **Autonomous Learning**
- **Self-directed research** using web resources
- **Continuous capability enhancement** through experience
- **Knowledge synthesis** from multiple sources
- **Adaptive tool creation** based on emerging needs

### **Collaborative AI Development**
- **Human-AI partnership** in software creation
- **Multi-agent coordination** for complex projects
- **Recursive self-improvement** capabilities
- **Dynamic architecture evolution**

## 🏃‍♂️ Development Workflow

### **Git Integration**
```bash
# Prepare changes on a new branch
./prepare-pull-request.sh feature/ai-improvements -x "py,js"

# Review AI-generated changes
git diff

# Complete the workflow
./complete-pull-request.sh
```

### **Autonomous Mode** (Experimental)
```python
# Enable autonomous learning cycles
orchestrator.start_autonomous_mode()
```

## 📊 Monitoring

AGIcommander includes built-in monitoring and metrics:

- **Task success rates** and performance tracking
- **Agent performance** analytics
- **Safety violation** monitoring
- **Resource usage** tracking
- **Learning progress** visualization

## 🤝 Contributing

AGIcommander welcomes contributions! See our [Contributing Guidelines](docs/CONTRIBUTING.md) for details.

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black .
flake8 .
```

## 📄 License

This project is dual-licensed:
- **Non-commercial use**: Free with attribution required
- **Commercial use**: Contact for licensing terms

See [LICENSE](LICENSE) for full details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

## 🙏 Acknowledgments

AGIcommander builds upon:
- **Model Context Protocol (MCP)** by Anthropic
- **LangChain** for LLM integration
- **ChromaDB** for vector storage
- The original **Commander** concept

---

**⚠️ Important**: AGIcommander includes experimental AGI features. Always review AI-generated changes before applying them to production systems.

**🔬 Research**: This project explores the boundaries of AI-assisted development and self-improving systems. Use responsibly and contribute to the understanding of safe AI development practices.
