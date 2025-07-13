To add all files in the current directory and everything below it to git, use:

```bash
git add .
```

That's it! The `.` means "current directory and all subdirectories recursively."

## **What `git add .` does:**
- ✅ **Adds all new files** in current directory and subdirectories
- ✅ **Stages all modified files** for commit
- ✅ **Respects .gitignore** - won't add ignored files
- ✅ **Recursive** - includes all nested directories

## **Alternative commands:**

```bash
# Add everything (same as git add .)
git add --all
git add -A

# Add only modified files (not new files)
git add -u

# Add specific file types
git add *.py
git add **/*.yaml

# Add specific directories
git add core/ servers/ agents/
```

## **Check what will be added:**
```bash
# See what files will be staged
git status

# See what changes will be staged
git diff --cached
```

## **For your AGIcommander project:**
```bash
# Add all project files
git add .

# Check what was staged
git status

# Commit everything
git commit -m "Initial AGIcommander project setup"
```

The `.gitignore` file we created will automatically exclude sensitive files like:
- `.env` (your API keys)
- `__pycache__/` (Python cache)
- `memory/` (AI data storage)
- `*.backup` (backup files)

So `git add .` is safe to use - it won't accidentally commit your secrets! 🔒

