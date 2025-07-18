# Here is a compact **BeautifulSoup Cheat Sheet** for Python 3:

---

## 🧰 **Setup**

```python
from bs4 import BeautifulSoup
import requests

html = requests.get("https://example.com").text
soup = BeautifulSoup(html, 'html.parser')  # or 'lxml', 'html5lib'
```

---

## 🔍 **Finding Elements**

| Expression                       | Purpose              |
| -------------------------------- | -------------------- |
| `soup.find('tag')`               | First `<tag>`        |
| `soup.find_all('tag')`           | All `<tag>` elements |
| `soup.find(id='some-id')`        | Tag with ID          |
| `soup.find(class_='some-class')` | Tag with class       |
| `soup.find_all('tag', limit=2)`  | First 2 tags         |
| `soup.select('div.classname')`   | CSS selector         |
| `soup.select_one('div#idname')`  | First match via CSS  |

---

## 📦 **Tag Attributes**

| Expression                         | Purpose                            |
| ---------------------------------- | ---------------------------------- |
| `tag.name`                         | Name of the tag (`div`, `a`, etc.) |
| `tag.attrs`                        | Dictionary of attributes           |
| `tag['href']` or `tag.get('href')` | Get attribute value                |

---

## 📝 **Text Content**

| Expression                     | Purpose                     |
| ------------------------------ | --------------------------- |
| `tag.text` or `tag.get_text()` | All text inside tag         |
| `tag.string`                   | Only if tag has 1 string    |
| `list(tag.stripped_strings)`   | Cleaned list of all strings |

---

## 🔄 **Navigation**

| Expression                 | Purpose                    |
| -------------------------- | -------------------------- |
| `tag.parent`               | Parent tag                 |
| `tag.parents`              | Generator of all ancestors |
| `tag.next_sibling`         | Next tag (or text node)    |
| `tag.previous_sibling`     | Previous tag               |
| `tag.find_next('tag')`     | Next matching tag          |
| `tag.find_previous('tag')` | Previous matching tag      |

---

## 🛠️ **Modify the DOM**

| Expression                  | Purpose               |
| --------------------------- | --------------------- |
| `tag.decompose()`           | Remove tag completely |
| `tag.extract()`             | Remove and return tag |
| `tag.append('text')`        | Add content to tag    |
| `tag.insert(0, new_tag)`    | Insert at position    |
| `tag.clear()`               | Remove all children   |
| `tag.replace_with(new_tag)` | Replace tag           |

---

## 🧪 **Example**

```python
html = "<html><body><a href='https://x.com'>X</a></body></html>"
soup = BeautifulSoup(html, 'html.parser')
link = soup.find('a')

print(link['href'])         # https://x.com
print(link.get_text())      # X
```

---

## 📚 Parser Options

| Parser          | Notes                                    |
| --------------- | ---------------------------------------- |
| `'html.parser'` | Built-in, fast                           |
| `'lxml'`        | Fast + tolerant (needs `lxml` installed) |
| `'html5lib'`    | Most forgiving (needs `html5lib`)        |

---

