The research server Python program should be placed at:

**`servers/learning/research.py`**

Here's the complete file structure:

```
AGIcommander/
├── servers/
│   ├── learning/
│   │   ├── research.py          ← Place the research server here
│   │   └── README.md            ← Place the README here (optional)
│   ├── memory/
│   │   ├── S3/
│   │   │   └── s3_mcp_server.py
│   │   └── vector_db/
│   │       └── vector_db_mcp_server.py
│   └── ...
```

## Why This Location?

**1. Matches Your Startup Configuration:**
Your `startup.yaml` references:
```yaml
learning:
  research:
    command: "python servers/learning/research.py"
```

**2. Follows Your Project Structure:**
- `servers/memory/` - Memory-related servers
- `servers/learning/` - Learning and research servers
- `servers/code/` - Code-related servers

**3. Matches pyproject.toml Entry Points:**
```toml
research-mcp-server = "servers.learning.research:main"
research_server = "servers.learning.research:ResearchMCPServer"
```

## Steps to Set It Up

```bash
# 1. Create the directory
mkdir -p servers/learning

# 2. Save the research server code as:
# servers/learning/research.py

# 3. Optionally save the README as:
# servers/learning/README.md

# 4. The server will auto-create its cache directory:
# memory/research_cache/
```

## Import Structure

With this placement, your code can import it as:
```python
from servers.learning.research import ResearchMCPServer
```

And your startup script will launch it with:
```bash
python servers/learning/research.py
```

This keeps everything organized according to your established architecture where each server type has its own directory under `servers/`!

